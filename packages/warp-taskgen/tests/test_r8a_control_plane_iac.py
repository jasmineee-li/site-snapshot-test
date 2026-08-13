from __future__ import annotations

import atexit
import os
import subprocess
from pathlib import Path

import yaml


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _local_host_config(repo_root: Path, tmp_path: Path) -> Path:
    """Create an ignored local overlay for subprocess tests that reach deploy validation."""
    path = repo_root / "configs" / "benchmark_hosts" / f"pytest-{tmp_path.name}.local.yaml"
    path.write_text((repo_root / "configs" / "benchmark_hosts" / "r8a.yaml").read_text())
    atexit.register(path.unlink, missing_ok=True)
    return path


class _CloudFormationLoader(yaml.SafeLoader):
    pass


def _construct_cloudformation_tag(loader: yaml.SafeLoader, tag_suffix: str, node: yaml.Node):
    if isinstance(node, yaml.ScalarNode):
        return {tag_suffix: loader.construct_scalar(node)}
    if isinstance(node, yaml.SequenceNode):
        return {tag_suffix: loader.construct_sequence(node)}
    if isinstance(node, yaml.MappingNode):
        return {tag_suffix: loader.construct_mapping(node)}
    raise TypeError(f"unsupported CloudFormation node: {node!r}")


_CloudFormationLoader.add_multi_constructor("!", _construct_cloudformation_tag)


def _load_template() -> dict:
    return yaml.load(
        (_repo_root() / "infra" / "cloudformation" / "r8a-control-plane.yaml").read_text(),
        Loader=_CloudFormationLoader,
    )


def test_r8a_control_plane_template_manages_eip_and_ssh_ingress() -> None:
    template = _load_template()

    resources = template["Resources"]
    assert resources["CanonicalElasticIp"]["Type"] == "AWS::EC2::EIP"
    assert resources["CanonicalElasticIpAssociation"]["Type"] == "AWS::EC2::EIPAssociation"
    assert resources["OperatorSshIngress1"]["Type"] == "AWS::EC2::SecurityGroupIngress"
    assert resources["OperatorSshIngress1"]["Properties"]["FromPort"] == 22
    assert resources["OperatorSshIngress1"]["Properties"]["ToPort"] == 22


def test_r8a_control_plane_template_has_no_world_open_ssh_default() -> None:
    template = _load_template()

    params = template["Parameters"]
    assert "Default" not in params["OperatorSshCidr1"]
    for name in ("OperatorSshCidr2", "OperatorSshCidr3", "OperatorSshCidr4", "OperatorSshCidr5"):
        assert params[name]["Default"] == ""

    rendered = str(template)
    assert "0.0.0.0/0" not in rendered
    assert "::/0" not in rendered


def test_deploy_r8a_control_plane_requires_local_host_config() -> None:
    repo_root = _repo_root()
    env = os.environ.copy()
    env["HOME"] = str(repo_root)

    completed = subprocess.run(
        [
            "bash",
            str(repo_root / "scripts" / "deploy_r8a_control_plane.sh"),
            "--operator-cidr",
            "203.0.113.10/32",
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "--host-config is required" in completed.stderr


def test_deploy_r8a_control_plane_refuses_public_template_for_deploy() -> None:
    repo_root = _repo_root()
    env = os.environ.copy()
    env["HOME"] = str(repo_root)

    completed = subprocess.run(
        [
            "bash",
            str(repo_root / "scripts" / "deploy_r8a_control_plane.sh"),
            "--host-config",
            "configs/benchmark_hosts/r8a.yaml",
            "--operator-cidr",
            "203.0.113.10/32",
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "refuses the tracked public template" in completed.stderr


def test_deploy_r8a_control_plane_refuses_write_to_public_template() -> None:
    repo_root = _repo_root()
    env = os.environ.copy()
    env["HOME"] = str(repo_root)

    completed = subprocess.run(
        [
            "bash",
            str(repo_root / "scripts" / "deploy_r8a_control_plane.sh"),
            "--host-config",
            "configs/benchmark_hosts/r8a.yaml",
            "--operator-cidr",
            "203.0.113.10/32",
            "--write-host-config",
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "refuses the tracked public template" in completed.stderr


def test_deploy_r8a_control_plane_rejects_world_open_operator_cidr(tmp_path: Path) -> None:
    repo_root = _repo_root()
    host_config = _local_host_config(repo_root, tmp_path)
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    aws = fakebin / "aws"
    aws.write_text("#!/bin/sh\nexit 0\n")
    aws.chmod(0o755)

    env = os.environ.copy()
    env["HOME"] = str(repo_root)
    env["PATH"] = f"{fakebin}:{env.get('PATH', '')}"

    completed = subprocess.run(
        [
            "bash",
            str(repo_root / "scripts" / "deploy_r8a_control_plane.sh"),
            "--host-config",
            str(host_config),
            "--operator-cidr",
            "0.0.0.0/0",
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "operator SSH CIDR must be a single IPv4 /32" in completed.stderr


def test_deploy_r8a_control_plane_rejects_broad_operator_cidr(tmp_path: Path) -> None:
    repo_root = _repo_root()
    host_config = _local_host_config(repo_root, tmp_path)
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    aws = fakebin / "aws"
    aws.write_text("#!/bin/sh\nexit 0\n")
    aws.chmod(0o755)

    env = os.environ.copy()
    env["HOME"] = str(repo_root)
    env["PATH"] = f"{fakebin}:{env.get('PATH', '')}"

    completed = subprocess.run(
        [
            "bash",
            str(repo_root / "scripts" / "deploy_r8a_control_plane.sh"),
            "--host-config",
            str(host_config),
            "--operator-cidr",
            "203.0.113.0/24",
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "operator SSH CIDR must be a single IPv4 /32" in completed.stderr


def test_deploy_r8a_control_plane_canonicalizes_bare_operator_ip(tmp_path: Path) -> None:
    repo_root = _repo_root()
    host_config = _local_host_config(repo_root, tmp_path)
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    aws = fakebin / "aws"
    calls = tmp_path / "aws.calls"
    aws.write_text(
        f"""#!/bin/sh
printf '%s\\n' "$*" >> {calls}
if [ "$1" = "ec2" ] && [ "$2" = "describe-addresses" ]; then
  printf 'None\\n'
  exit 0
fi
if [ "$1" = "cloudformation" ] && [ "$2" = "deploy" ]; then
  exit 0
fi
if [ "$1" = "cloudformation" ] && [ "$2" = "describe-stacks" ]; then
  printf '[{{"OutputKey":"ElasticIp","OutputValue":""}},{{"OutputKey":"AllocationId","OutputValue":"eipalloc-0123456789abcdef0"}},{{"OutputKey":"InstanceId","OutputValue":"i-0123456789abcdef0"}}]\\n'
  exit 0
fi
if [ "$1" = "ec2" ] && [ "$2" = "describe-instances" ]; then
  printf '198.51.100.40\\n'
  exit 0
fi
printf 'unexpected aws call: %s\\n' "$*" >&2
exit 99
"""
    )
    aws.chmod(0o755)

    env = os.environ.copy()
    env["HOME"] = str(repo_root)
    env["PATH"] = f"{fakebin}:{env.get('PATH', '')}"

    completed = subprocess.run(
        [
            "bash",
            str(repo_root / "scripts" / "deploy_r8a_control_plane.sh"),
            "--host-config",
            str(host_config),
            "--operator-cidr",
            "203.0.113.10",
            "--instance-id",
            "i-0123456789abcdef0",
            "--existing-allocation-id",
            "eipalloc-0123456789abcdef0",
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "OperatorSshCidr1=203.0.113.10/32" in calls.read_text()


def test_deploy_r8a_control_plane_refuses_to_move_attached_eip_without_force(
    tmp_path: Path,
) -> None:
    repo_root = _repo_root()
    host_config = _local_host_config(repo_root, tmp_path)
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    aws = fakebin / "aws"
    aws.write_text(
        """#!/bin/sh
if [ "$1" = "ec2" ] && [ "$2" = "describe-addresses" ]; then
  printf 'i-0123456789abcdef1\\n'
  exit 0
fi
printf 'unexpected aws call: %s\\n' "$*" >&2
exit 99
"""
    )
    aws.chmod(0o755)

    env = os.environ.copy()
    env["HOME"] = str(repo_root)
    env["PATH"] = f"{fakebin}:{env.get('PATH', '')}"

    completed = subprocess.run(
        [
            "bash",
            str(repo_root / "scripts" / "deploy_r8a_control_plane.sh"),
            "--host-config",
            str(host_config),
            "--operator-cidr",
            "203.0.113.10/32",
            "--instance-id",
            "i-0123456789abcdef0",
            "--existing-allocation-id",
            "eipalloc-0123456789abcdef1",
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "is attached to i-0123456789abcdef1" in completed.stderr
    assert "--allow-reassociate-existing-eip" in completed.stderr


def test_deploy_r8a_control_plane_accepts_attached_eip_with_force(tmp_path: Path) -> None:
    repo_root = _repo_root()
    host_config = _local_host_config(repo_root, tmp_path)
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    aws = fakebin / "aws"
    calls = tmp_path / "aws.calls"
    aws.write_text(
        f"""#!/bin/sh
printf '%s\\n' "$*" >> {calls}
if [ "$1" = "ec2" ] && [ "$2" = "describe-addresses" ]; then
  printf 'i-0123456789abcdef1\\n'
  exit 0
fi
if [ "$1" = "cloudformation" ] && [ "$2" = "deploy" ]; then
  exit 0
fi
if [ "$1" = "cloudformation" ] && [ "$2" = "describe-stacks" ]; then
  printf '[{{"OutputKey":"ElasticIp","OutputValue":""}},{{"OutputKey":"AllocationId","OutputValue":"eipalloc-0123456789abcdef1"}},{{"OutputKey":"InstanceId","OutputValue":"i-0123456789abcdef0"}}]\\n'
  exit 0
fi
if [ "$1" = "ec2" ] && [ "$2" = "describe-instances" ]; then
  printf '198.51.100.40\\n'
  exit 0
fi
printf 'unexpected aws call: %s\\n' "$*" >&2
exit 99
"""
    )
    aws.chmod(0o755)

    env = os.environ.copy()
    env["HOME"] = str(repo_root)
    env["PATH"] = f"{fakebin}:{env.get('PATH', '')}"

    completed = subprocess.run(
        [
            "bash",
            str(repo_root / "scripts" / "deploy_r8a_control_plane.sh"),
            "--host-config",
            str(host_config),
            "--operator-cidr",
            "203.0.113.10/32",
            "--instance-id",
            "i-0123456789abcdef0",
            "--existing-allocation-id",
            "eipalloc-0123456789abcdef1",
            "--allow-reassociate-existing-eip",
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert (
        "WARNING: moving EIP eipalloc-0123456789abcdef1 from i-0123456789abcdef1 to i-0123456789abcdef0"
        in completed.stderr
    )
    assert "public_ip:   198.51.100.40" in completed.stdout
    deploy_call = calls.read_text()
    assert "cloudformation deploy" in deploy_call
    assert "--capabilities CAPABILITY_IAM" in deploy_call


def test_audit_r8a_control_plane_passes_when_eip_stack_and_config_match(tmp_path: Path) -> None:
    repo_root = _repo_root()
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    aws = fakebin / "aws"
    aws.write_text(
        """#!/bin/sh
if [ "$1" = "ec2" ] && [ "$2" = "describe-instances" ]; then
  printf 'i-0123456789abcdef0\\n'
  exit 0
fi
if [ "$1" = "ec2" ] && [ "$2" = "describe-network-interfaces" ]; then
  printf '{"PublicIp":"203.0.113.10","AllocationId":"eipalloc-0123456789abcdef0","Groups":["sg-0123456789abcdef0"]}\\n'
  exit 0
fi
if [ "$1" = "ec2" ] && [ "$2" = "describe-security-group-rules" ]; then
  printf '[{"IpProtocol":"tcp","FromPort":22,"ToPort":22,"CidrIpv4":"203.0.113.10/32"}]\\n'
  exit 0
fi
if [ "$1" = "cloudformation" ] && [ "$2" = "describe-stacks" ]; then
  printf '{"Outputs":[{"OutputKey":"AllocationId","OutputValue":"eipalloc-0123456789abcdef0"},{"OutputKey":"SecurityGroupId","OutputValue":"sg-0123456789abcdef0"}],"Parameters":[{"ParameterKey":"OperatorSshCidr1","ParameterValue":"203.0.113.10/32"}]}\\n'
  exit 0
fi
printf 'unexpected aws call: %s\\n' "$*" >&2
exit 99
"""
    )
    aws.chmod(0o755)

    env = os.environ.copy()
    env["HOME"] = str(repo_root)
    env["PATH"] = f"{fakebin}:{env.get('PATH', '')}"

    completed = subprocess.run(
        ["bash", str(repo_root / "scripts" / "audit_r8a_control_plane.sh")],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "r8a_control_plane=ok" in completed.stdout


def test_audit_r8a_control_plane_fails_when_managed_ssh_rule_missing(tmp_path: Path) -> None:
    repo_root = _repo_root()
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    aws = fakebin / "aws"
    aws.write_text(
        """#!/bin/sh
if [ "$1" = "ec2" ] && [ "$2" = "describe-instances" ]; then
  printf 'i-0123456789abcdef0\\n'
  exit 0
fi
if [ "$1" = "ec2" ] && [ "$2" = "describe-network-interfaces" ]; then
  printf '{"PublicIp":"198.51.100.10","AllocationId":"eipalloc-0123456789abcdef0","Groups":["sg-0123456789abcdef0"]}\\n'
  exit 0
fi
if [ "$1" = "ec2" ] && [ "$2" = "describe-security-group-rules" ]; then
  printf '[]\\n'
  exit 0
fi
if [ "$1" = "cloudformation" ] && [ "$2" = "describe-stacks" ]; then
  printf '{"Outputs":[{"OutputKey":"AllocationId","OutputValue":"eipalloc-0123456789abcdef0"},{"OutputKey":"SecurityGroupId","OutputValue":"sg-0123456789abcdef0"}],"Parameters":[{"ParameterKey":"OperatorSshCidr1","ParameterValue":"203.0.113.10/32"}]}\\n'
  exit 0
fi
printf 'unexpected aws call: %s\\n' "$*" >&2
exit 99
"""
    )
    aws.chmod(0o755)

    env = os.environ.copy()
    env["HOME"] = str(repo_root)
    env["PATH"] = f"{fakebin}:{env.get('PATH', '')}"

    completed = subprocess.run(
        ["bash", str(repo_root / "scripts" / "audit_r8a_control_plane.sh")],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "security group is missing managed SSH CIDRs: 203.0.113.10/32" in completed.stderr


def test_audit_r8a_control_plane_fails_on_world_open_ssh_rule(tmp_path: Path) -> None:
    repo_root = _repo_root()
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    aws = fakebin / "aws"
    aws.write_text(
        """#!/bin/sh
if [ "$1" = "ec2" ] && [ "$2" = "describe-instances" ]; then
  printf 'i-0123456789abcdef0\\n'
  exit 0
fi
if [ "$1" = "ec2" ] && [ "$2" = "describe-network-interfaces" ]; then
  printf '{"PublicIp":"198.51.100.10","AllocationId":"eipalloc-0123456789abcdef0","Groups":["sg-0123456789abcdef0"]}\\n'
  exit 0
fi
if [ "$1" = "ec2" ] && [ "$2" = "describe-security-group-rules" ]; then
  printf '[{"IpProtocol":"tcp","FromPort":22,"ToPort":22,"CidrIpv4":"203.0.113.10/32"},{"IpProtocol":"tcp","FromPort":22,"ToPort":22,"CidrIpv4":"0.0.0.0/0"}]\\n'
  exit 0
fi
if [ "$1" = "cloudformation" ] && [ "$2" = "describe-stacks" ]; then
  printf '{"Outputs":[{"OutputKey":"AllocationId","OutputValue":"eipalloc-0123456789abcdef0"},{"OutputKey":"SecurityGroupId","OutputValue":"sg-0123456789abcdef0"}],"Parameters":[{"ParameterKey":"OperatorSshCidr1","ParameterValue":"203.0.113.10/32"}]}\\n'
  exit 0
fi
printf 'unexpected aws call: %s\\n' "$*" >&2
exit 99
"""
    )
    aws.chmod(0o755)

    env = os.environ.copy()
    env["HOME"] = str(repo_root)
    env["PATH"] = f"{fakebin}:{env.get('PATH', '')}"

    completed = subprocess.run(
        ["bash", str(repo_root / "scripts" / "audit_r8a_control_plane.sh")],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "security group has world-open SSH ingress" in completed.stderr
