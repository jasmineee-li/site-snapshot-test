# Issue 3: Restructure Directories to Group Benign + Adversarial Variants

## Problem Statement

Currently, benign and adversarial variants are in separate top-level directories:
```
results/study/
├── date_agent_behavior.benign_0/          # Separate, hard to pair
└── date_agent_behavior.adversarial_v0_0/  # Separate, hard to pair
```

**Desired structure**: Group all variants of a behavior under one parent directory:
```
results/study/
└── date_agent_behavior_seed0/             # Single logical "run"
    ├── metadata.json                      # Shared: behavior config, variants list
    ├── base/                              # Shared: base HTML before injections
    ├── benign/                            # Variant 1: benign run
    │   ├── *.html                         # Benign HTML files
    │   ├── summary_info.json
    │   ├── judge_result.json
    │   └── step_*.pkl.gz
    └── adversarial_v0/                    # Variant 2: adversarial run
        ├── *.html                         # Adversarial HTML files
        ├── summary_info.json
        ├── judge_result.json
        └── step_*.pkl.gz
```

## Key Insight: One "Behavior Run" = Multiple Variant Subdirectories

Instead of treating benign and adversarial as separate tasks, treat them as **variants of one logical behavior run**.

**AgentLab Compatibility**: Still use one exp_dir per execution, but nest variant-specific files in subdirectories while sharing parent-level metadata.

---

## Implementation Strategy

### Core Principle
- **Study level**: One "behavior run" distributed as multiple Ray tasks (one per variant)
- **Task level**: Each variant knows its parent directory and variant name
- **Storage**: Variant-specific files go in subdirectories, shared files in parent
- **Loading**: Result loading detects parent dirs and enumerates variant subdirectories

---

## Detailed Plan

### Phase 1: Modify RedteamEnvArgs to Include Parent Context

**File**: `AgentLab/src/agentlab/benchmarks/redteam.py`

**Add fields to RedteamEnvArgs** (around line 64-134):
```python
@dataclass
class RedteamEnvArgs(AbstractEnvArgs):
    # ... existing fields ...

    # New: Parent directory tracking
    parent_exp_dir: str = ""           # Parent dir grouping all variants
    variant_name: str = ""             # "benign" or "adversarial_v0"
    is_variant_run: bool = False       # True if using variant structure

    @property
    def variant_subdir(self) -> str:
        """Subdirectory name for this variant."""
        if self.test_mode == "benign":
            return "benign"
        else:
            return f"adversarial_v{self.variation_seed}"
```

**Why**: Need to pass parent directory context through the task distribution system.

---

### Phase 2: Modify Study to Pre-Create Parent Directories

**File**: `AgentLab/src/agentlab/experiments/study.py`

**Approach**: Before launching Ray tasks, group variants by behavior and create parent directories.

**Add method to Study class**:
```python
def _prepare_redteam_parent_dirs(
    self,
    benchmark,
    exp_root: Path,
    date_str: str,
) -> Dict[tuple, Path]:
    """
    Pre-create parent directories for redteam behavior runs.

    Groups variants by (behavior_id, variation_seed) and creates parent dirs.

    Returns:
        Dict mapping (behavior_id, seed) -> parent_exp_dir
    """
    from collections import defaultdict

    # Check if this is redteam benchmark
    if not hasattr(benchmark, 'env_args_list'):
        return {}

    # Group tasks by (behavior_id, variation_seed)
    behavior_groups = defaultdict(list)
    for env_args in benchmark.env_args_list:
        if hasattr(env_args, 'behavior_id'):
            key = (env_args.behavior_id, env_args.variation_seed)
            behavior_groups[key].append(env_args)

    # Create parent directory for each behavior group
    parent_dirs = {}
    for (behavior_id, seed), env_args_list in behavior_groups.items():
        # Create parent directory name
        agent_name = self.agent_args.agent_name
        parent_name = f"{date_str}_{agent_name}_{behavior_id}_seed{seed}"
        parent_dir = exp_root / parent_name
        parent_dir.mkdir(parents=True, exist_ok=True)

        # Save metadata about this behavior run
        metadata = {
            "behavior_id": behavior_id,
            "variation_seed": seed,
            "variants": [
                env_args.variant_subdir for env_args in env_args_list
            ],
            "created_at": date_str,
        }
        (parent_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

        # Store for later reference
        parent_dirs[(behavior_id, seed)] = parent_dir

        logger.info(f"Created parent dir for {behavior_id} seed{seed}: {parent_dir}")

    return parent_dirs
```

**Modify Study.run()** (around line 150-250):
```python
def run(self, benchmark, agent_args_list, n_jobs=1, ...):
    # ... existing setup ...

    # For redteam: pre-create parent directories
    parent_dirs = {}
    if isinstance(benchmark, RedteamBenchmark):
        parent_dirs = self._prepare_redteam_parent_dirs(
            benchmark, exp_root, date_str
        )

        # Update env_args with parent directory info
        for env_args in benchmark.env_args_list:
            if hasattr(env_args, 'behavior_id'):
                key = (env_args.behavior_id, env_args.variation_seed)
                if key in parent_dirs:
                    env_args.parent_exp_dir = str(parent_dirs[key])
                    env_args.variant_name = env_args.variant_subdir
                    env_args.is_variant_run = True

    # Continue with normal task distribution
    # ...
```

**Why**: Parent directories must exist before variants run. Study is the right place since it orchestrates all tasks.

---

### Phase 3: Modify RedteamEnv to Use Variant Subdirectories

**File**: `AgentLab/src/agentlab/benchmarks/redteam.py`

**Modify `_generate_sites()` to use parent/variant structure** (around line 400-720):

```python
def _generate_sites(self, exp_dir: Path, behavior_config: dict):
    """Generate synthetic websites with injection variants."""

    # Determine directory structure
    if self.env_args.is_variant_run and self.env_args.parent_exp_dir:
        # New structure: parent/variant/
        parent_dir = Path(self.env_args.parent_exp_dir)
        variant_name = self.env_args.variant_name

        # Set variant-specific directory
        self.exp_dir = parent_dir / variant_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        # Base HTML goes in parent/base/ (shared across variants)
        base_dir = parent_dir / "base"

        # Shared metadata goes in parent/
        shared_metadata_dir = parent_dir

        logger.info(f"Using variant structure: {parent_dir}/{variant_name}")
    else:
        # Old structure: backward compatibility
        self.exp_dir = exp_dir
        base_dir = self.exp_dir / "base"
        shared_metadata_dir = self.exp_dir

        logger.info(f"Using legacy structure: {exp_dir}")

    base_dir.mkdir(parents=True, exist_ok=True)

    # ... rest of generation logic ...

    # Save base HTML to shared base/ directory
    for file_name, html in base_html_by_subdomain.items():
        (base_dir / f"{file_name}.html").write_text(html, encoding="utf-8")

    # Save placeholder content to parent (shared)
    placeholder_content_path = shared_metadata_dir / "placeholder_content.json"
    # ... save logic ...

    # Save injection log to parent (shared)
    injection_log_path = shared_metadata_dir / "injections.json"
    # ... save logic ...

    # Save variant-specific HTML to variant subdirectory
    variant_dir = self.exp_dir  # Already set above
    for file_name, html in variant_html.items():
        (variant_dir / f"{file_name}.html").write_text(html, encoding="utf-8")

    # Flow config points to variant directory
    self._flow_config["run_dir"] = str(variant_dir)
    config_path = self.exp_dir / "flow_config.json"
    config_path.write_text(json.dumps(self._flow_config, indent=2))
```

**Why**: This keeps the generation logic mostly the same, just changes where files are saved.

---

### Phase 4: Modify ExperimentLoop to Handle Variant Directories

**File**: `AgentLab/src/agentlab/experiments/loop.py`

**Modify exp_dir creation** (around line 380-410):

```python
def __init__(self, env_args, agent_args, ...):
    # ... existing init ...

    # Check if using variant structure
    if hasattr(env_args, 'is_variant_run') and env_args.is_variant_run:
        # Parent already created by Study
        parent_dir = Path(env_args.parent_exp_dir)
        variant_name = env_args.variant_name

        # Use parent/variant as exp_dir
        self.exp_dir = parent_dir / variant_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        # Store parent reference for accessing shared files
        self._parent_exp_dir = parent_dir

        logger.info(f"Using variant exp_dir: {self.exp_dir}")
    else:
        # Normal exp_dir creation
        # ... existing logic ...
```

**Add property to access parent directory**:
```python
@property
def parent_exp_dir(self) -> Optional[Path]:
    """Parent directory if using variant structure."""
    return getattr(self, '_parent_exp_dir', None)
```

**Why**: ExperimentLoop needs to know about the variant structure to save files correctly.

---

### Phase 5: Update Result Loading to Handle Variant Directories

**File**: `AgentLab/src/agentlab/experiments/loop.py`

**Modify `ExpResult` class**:

```python
class ExpResult:
    def __init__(self, exp_dir) -> None:
        self.exp_dir = Path(exp_dir)

        # Detect if this is a variant subdirectory
        self.parent_dir = None
        self.variant_name = None

        # Check if parent has metadata.json
        if (self.exp_dir.parent / "metadata.json").exists():
            try:
                with open(self.exp_dir.parent / "metadata.json") as f:
                    metadata = json.load(f)
                    if "variants" in metadata:
                        self.parent_dir = self.exp_dir.parent
                        self.variant_name = self.exp_dir.name
                        logger.debug(f"Detected variant: {self.variant_name} in {self.parent_dir}")
            except Exception as e:
                logger.warning(f"Failed to load parent metadata: {e}")

        # ... rest of init ...

    @property
    def is_variant_run(self) -> bool:
        """Check if this is a variant subdirectory."""
        return self.parent_dir is not None

    @property
    def shared_metadata_dir(self) -> Path:
        """Directory for shared metadata (parent if variant, else exp_dir)."""
        return self.parent_dir if self.is_variant_run else self.exp_dir
```

**Modify `get_exp_record()` to include variant info**:
```python
def get_exp_record(self) -> dict:
    record = {"exp_dir": self.exp_dir}

    # Add variant info if applicable
    if self.is_variant_run:
        record["parent_exp_dir"] = str(self.parent_dir)
        record["variant_name"] = self.variant_name

    # ... rest of method ...
```

**Why**: Result loading needs to understand the new structure to find all files.

---

### Phase 6: Update yield_all_exp_results to Find Variants

**File**: `AgentLab/src/agentlab/experiments/loop.py`

**Modify `yield_all_exp_results()`** (around line 1164-1220):

```python
def yield_all_exp_results(
    savedir_base: str | Path,
    progress_fn=tqdm,
    load_hidden=False,
    use_cache=True
):
    """
    Recursively find all experiments, including variant subdirectories.

    Detects both:
    - Standard exp_dirs (old structure)
    - Variant subdirectories (new structure: parent/benign/, parent/adversarial_v*/)
    """
    for path in Path(savedir_base).rglob("*"):
        if not path.is_dir():
            continue

        # Skip hidden directories
        if not load_hidden and (path.name.startswith("_") or path.name.startswith(".")):
            continue

        # Check for variant parent directory
        if (path / "metadata.json").exists():
            metadata_file = path / "metadata.json"
            try:
                with open(metadata_file) as f:
                    metadata = json.load(f)
                    if "variants" in metadata:
                        # This is a parent dir - enumerate variants
                        logger.debug(f"Found variant parent: {path}")
                        for variant_name in metadata["variants"]:
                            variant_dir = path / variant_name
                            if variant_dir.exists() and (variant_dir / "exp_args.pkl").exists():
                                logger.debug(f"  - Yielding variant: {variant_name}")
                                yield get_exp_result(variant_dir) if use_cache else ExpResult(variant_dir)
                        continue  # Don't process this dir further
            except Exception as e:
                logger.warning(f"Failed to process metadata in {path}: {e}")

        # Check for standard exp_dir (backward compatibility)
        if (path / "exp_args.pkl").exists():
            # Make sure this isn't a variant subdir we already processed
            parent_metadata = path.parent / "metadata.json"
            if parent_metadata.exists():
                # Skip - already handled above
                continue

            logger.debug(f"Found standard exp_dir: {path}")
            yield get_exp_result(path) if use_cache else ExpResult(path)
```

**Why**: Result loading must find both old-style exp_dirs and new variant subdirectories.

---

### Phase 7: Update JudgeRunner to Load Metadata from Parent

**File**: `AgentLab/src/agentlab/benchmarks/redteam_judge.py`

**Modify `_load_injection_metadata()`** (around line 662):

```python
def _load_injection_metadata(self, exp_dir: Path) -> Dict[str, Any]:
    """Load injection metadata, checking both exp_dir and parent_dir."""

    # Try exp_dir first (old structure or variant-specific)
    injection_file = Path(exp_dir) / "injections.json"

    # If not found, try parent dir (new structure)
    if not injection_file.exists():
        parent_metadata = Path(exp_dir).parent / "metadata.json"
        if parent_metadata.exists():
            try:
                with open(parent_metadata) as f:
                    metadata = json.load(f)
                    if "variants" in metadata:
                        # This is a variant - check parent for injections.json
                        injection_file = Path(exp_dir).parent / "injections.json"
            except Exception:
                pass

    if not injection_file.exists():
        logger.warning(f"injections.json not found in {exp_dir} or parent")
        return {
            "test_mode": "unknown",
            "injection_locations": [],
            "placeholder_content": {},
        }

    # ... rest of method unchanged ...
```

**Why**: Judge needs to find injection metadata whether in variant subdir or parent.

---

## Migration Strategy

### Backward Compatibility

**All changes are backward compatible**:
- Old exp_dirs without `is_variant_run=True` continue to work as before
- Result loading checks for both structures
- Analysis functions work with both structures

### Transition Plan

**Phase 1: Implement with feature flag**
```python
# In RedteamEnvArgs
use_variant_structure: bool = True  # New default

# In Study.run()
if use_variant_structure and isinstance(benchmark, RedteamBenchmark):
    # Use new structure
else:
    # Use old structure
```

**Phase 2: Run experiments with both structures**
- Verify new structure works
- Compare analysis results
- Test all scripts (analyze_results, agent_xray, etc.)

**Phase 3: Make new structure default**
- Set `use_variant_structure=True` as default
- Update documentation
- Deprecate old structure

**Phase 4: Clean up old code**
- Remove backward compatibility code
- Update all examples to use new structure

---

## Testing Checklist

### Unit Tests
- [ ] Parent directory creation in Study
- [ ] Variant subdirectory structure in RedteamEnv
- [ ] File saving to correct locations (base/, variant/)
- [ ] Result loading finds all variants
- [ ] Judge loads metadata from parent

### Integration Tests
- [ ] Run full redteam experiment with new structure
- [ ] Verify directory layout matches spec
- [ ] Load results and verify all fields present
- [ ] Run analyze_results() on new structure
- [ ] Run global_report() on new structure
- [ ] Agent X-Ray works with new structure

### Backward Compatibility Tests
- [ ] Load old-structure results
- [ ] Mix old and new results in same study
- [ ] All analysis functions work with both

### Edge Cases
- [ ] Empty parent directory (no variants completed)
- [ ] Partial completion (only benign finished)
- [ ] Multiple seeds for same behavior
- [ ] Nested result directories

---

## Benefits of New Structure

### For Users
1. **Easy pairing**: All variants in one directory
2. **Shared metadata**: One place for behavior config, base HTML
3. **Clear organization**: Obvious which runs belong together
4. **Disk space**: Base HTML not duplicated per variant

### For Analysis
1. **Simpler grouping**: Just look at parent directory contents
2. **Metadata access**: All shared data in parent
3. **Comparison**: Variants side-by-side

### For Development
1. **Cleaner logic**: One behavior = one parent directory
2. **Better caching**: Can cache base HTML at parent level
3. **Easier debugging**: Related files co-located

---

## Estimated Effort

| Task | Files | Lines | Complexity | Time |
|------|-------|-------|------------|------|
| Study parent dir creation | study.py | ~80 | Medium | 2h |
| RedteamEnv variant structure | redteam.py | ~100 | Medium | 3h |
| ExperimentLoop variant handling | loop.py | ~50 | Low | 1h |
| Result loading updates | loop.py | ~100 | High | 4h |
| Judge metadata parent lookup | redteam_judge.py | ~20 | Low | 30m |
| Testing | - | - | High | 4h |
| **Total** | **5 files** | **~350 lines** | - | **~15h** |

---

## Alternative: Minimal Post-Processing Approach

If full restructure is too complex, consider **post-processing symlinks**:

### Quick Implementation
```python
# scripts/organize_redteam_results.py
def organize_results(study_dir):
    """Create parent directories with symlinks to variants."""

    # Find all benign/adversarial pairs
    results = load_result_df(study_dir)
    df = results.reset_index(inplace=False)

    # Group by behavior
    for (behavior_id, seed), group in df.groupby(['env.behavior_id', 'env.variation_seed']):
        # Create parent directory
        parent_dir = study_dir / f"grouped_{behavior_id}_seed{seed}"
        parent_dir.mkdir(exist_ok=True)

        # Create symlinks
        for _, row in group.iterrows():
            exp_dir = Path(row['exp_dir'])
            variant_name = exp_dir.name.split('.')[-2]  # Extract benign/adversarial_v0
            symlink = parent_dir / variant_name
            if not symlink.exists():
                symlink.symlink_to(exp_dir, target_is_directory=True)

        # Create metadata
        metadata = {
            "behavior_id": behavior_id,
            "seed": seed,
            "variants": list(parent_dir.iterdir()),
        }
        (parent_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
```

**Pros**:
- No code changes to core AgentLab
- Can run on existing results
- Reversible

**Cons**:
- Doesn't save disk space (base HTML still duplicated)
- Symlinks may not work on all systems
- Still have redundant directories

---

## Recommendation

**Implement the full restructure** (not post-processing):
1. More robust and maintainable
2. Saves disk space (shared base HTML)
3. Better UX from the start
4. Cleaner for future features

**Start with Phase 1-3** (directory structure):
- These have the most user impact
- Less risky than result loading changes
- Can validate structure before updating analysis

**Then add Phase 4-7** (result loading):
- Test thoroughly with backward compatibility
- Roll out gradually with feature flag
