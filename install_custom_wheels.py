from typing import Iterable, List, Tuple

from dependency_installer import (
    TRELLIS_DEPENDENCIES,
    check_package_installed,
    create_venv,
    get_installation_status,
    get_venv_path,
    install_custom_wheels,
    install_package,
    install_pytorch,
    install_triton,
    verify_installation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the TRELLIS venv, install CUDA 12.4 wheels, and verify imports.",
    )
    parser.add_argument(
        "--force-recreate",
        action="store_true",
        help="Delete the existing venv before provisioning",
    )
    parser.add_argument(
        "--skip-wheel-download",
        action="store_true",
        help="Skip fetching HuggingFace wheels (expects files in Documents/wheels)",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Non-interactive mode (assume yes to prompts)",
    )
    return parser.parse_args()


def confirm(prompt: str, auto_yes: bool) -> None:
    if auto_yes:
        return
    answer = input(f"{prompt} (yes/no): ").strip().lower()
    if answer != "yes":
        raise SystemExit("Aborted by user")


def upgrade_bootstrap() -> None:
    for package in ("pip", "setuptools", "wheel"):
        ok, msg = install_package(package, upgrade=True)
        if not ok:
            raise RuntimeError(f"Failed to upgrade {package}: {msg}")


def install_dependency_batch(packages: Iterable[str]) -> Tuple[int, List[str]]:
    failed: List[str] = []
    installed = 0
    for spec in packages:
        pkg_name = spec.split("[")[0].split("==")[0]
        already, version = check_package_installed(pkg_name)
        if already:
            print(f"✓ {pkg_name} ({version})")
            installed += 1
            continue
        print(f"→ Installing {spec}")
        ok, msg = install_package(spec)
        if ok:
            installed += 1
        else:
            print(f"  ✗ {msg}")
            failed.append(spec)
    return installed, failed


def main() -> None:
    args = parse_args()
    venv_path = get_venv_path()

    print(f"Virtual environment: {venv_path}")
    if args.force_recreate and venv_path.exists():
        confirm("Remove existing venv?", args.yes)
        shutil.rmtree(venv_path)
        print("✓ Removed old venv")

    confirm("Proceed with dependency installation?", args.yes)

    success, msg = create_venv()
    if not success:
        raise SystemExit(msg)
    print(msg)

    print("\nUpgrading pip/setuptools/wheel")
    upgrade_bootstrap()

    print("\nInstalling PyTorch 2.6.0 + CUDA 12.4")
    ok, msg = install_pytorch()
    if not ok:
        raise SystemExit(msg)
    print(msg)

    print("\nInstalling Triton compiler")
    ok, msg = install_triton()
    if not ok:
        print(f"⚠ Triton installation failed: {msg}")
    else:
        print(msg)

    if not args.skip_wheel_download:
        print("\nInstalling custom CUDA wheels")
        wheel_ok, wheel_msgs = install_custom_wheels()
        for entry in wheel_msgs:
            print(f"  {entry}")
        if not wheel_ok:
            print("⚠ Some custom wheels failed; review output above")
    else:
        print("\nSkipping HuggingFace wheel download as requested")

    print("\nInstalling Python dependencies")
    installed, failed = install_dependency_batch(TRELLIS_DEPENDENCIES)
    total = len(TRELLIS_DEPENDENCIES)
    print(f"Installed {installed}/{total} packages")
    if failed:
        print("⚠ Failed packages:")
        for item in failed:
            print(f"  - {item}")

    print("\nVerifying environment")
    ok, messages = verify_installation()
    for message in messages:
        print(f"  {message}")

    status = get_installation_status(detailed=False)
    print("\nSummary:")
    for key, value in status.items():
        print(f"  {key}: {value}")

    if not ok or failed:
        raise SystemExit("Installation completed with issues; see log above")
    print("\n✅ Environment ready! Activate it via Scripts/Activate.ps1 inside TRELLIS_venv.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)
