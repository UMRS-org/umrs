# UMRS Configuration Examples

This directory contains example JSON configuration files for UMRS server and
user setups. These files are references and starting points; they are **not**
deployed automatically.

The older `deploy_cfg.sh` helper is no longer part of the current branch.
Today, configuration is usually installed by site-specific setup scripts such
as `deploy/rccs/setup_client.sh`, or copied manually into `~/.config/umrs/`.

## Typical Target Layout

```text
~/.config/umrs/
  conf_umrs.json
  users/
    user1.json
    user2.json
    spring8.json
  tmp/
```

## What These Examples Cover

- `conf_umrs*.json` files show server-side configuration patterns.
- Server configs can declare `user_files_dir`; uploaded/reloaded user JSONs are
  stored there instead of relying on an implicit default next to the server.
- `umrs/users/*.json` files show per-user experiences, bucket aliases,
  modality configuration, and model settings.
- The examples include both general-purpose users and more specialized
  tomography/HMAE-oriented configurations.

## Recommended Usage

### Manual copy

Copy the example files you want to adapt into your config directory and edit
them there:

```bash
mkdir -p ~/.config/umrs/users
cp src/umrs_cfg_example/umrs/conf_umrs.json ~/.config/umrs/conf_umrs.json
cp src/umrs_cfg_example/umrs/users/user1.json ~/.config/umrs/users/user1.json
```

The example `conf_umrs.json` already sets `"user_files_dir": "~/.config/umrs/users"`
so server-side FILE uploads and reloads keep using the same canonical user
config directory.

### Site setup scripts

For real deployments, prefer the setup scripts under `deploy/` because they
install configs that match the deployment wrapper, tunnel, and site defaults.

## See Also

- [../../doc/umrs_overview.md](../../doc/umrs_overview.md)
- [../../doc/multi_users.md](../../doc/multi_users.md)
- [../../deploy/rccs-front/README.md](../../deploy/rccs-front/README.md)

