import pathlib

files = [
    pathlib.Path.home() / "ai4s-platform/umrs/deploy/bdrt/serve-model.slurm",
    pathlib.Path.home() / "ai4s-platform/umrs/src/umrs/mini-rivault-llm/serve-model.slurm",
]

for fpath in files:
    print(f"Fixing {fpath.name}...")
    text = fpath.read_text()
    
    # Remove [containers] section from storage.conf heredoc
    text = text.replace("[containers]\n" + 'isolation = "chroot"\n', "")
    text = text.replace("[containers]\n" + 'isolation = "chroot"\n\n', "")
    
    # Add BUILDAH_ISOLATION after storage.conf EOF block
    old = "EOF\n\nuuid=$(cat /proc/sys/kernel/random/uuid)"
    new = """EOF

# Chroot isolation (user-NS GID mapping gap on compute nodes)
export BUILDAH_ISOLATION=chroot
export CONTAINERS_BUILDAH_ISOLATION=chroot

uuid=$(cat /proc/sys/kernel/random/uuid)"""
    text = text.replace(old, new, 1)
    
    # Fix duplicate mkdir
    text = text.replace(
        'mkdir -p "${HOME}/.config/containers/"\n\nmkdir -p "${HOME}/.config/containers/"\n',
        'mkdir -p "${HOME}/.config/containers/"\n'
    )
    
    fpath.write_text(text)
    
    assert "[containers]" not in fpath.read_text(), f"[containers] still present in {fpath.name}"
    assert "BUILDAH_ISOLATION=chroot" in fpath.read_text(), f"BUILDAH_ISOLATION not in {fpath.name}"
    print(f"  OK {fpath.name}")

print("Done!")
