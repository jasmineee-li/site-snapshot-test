The {variant} sanity check for the current app directory failed with the following output:

```
{output}
```

Fix the issues so the sanity check passes. The current working directory is already the app root. Read `{verifier_sanity_check_guide_path}` for the sanity-check contract, inspect `./sanity_check_{variant}.py`, and then fix the app code and/or verifier tasks to make it pass. Do not create or reference a nested `apps/{app_id}` directory. Do not modify `server.py`; it is a reserved runtime file.
