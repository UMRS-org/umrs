#!/usr/bin/python3
import json
import os
import sys
from umrs.sql import *
from umrs.s3 import *
from umrs.MessageQueue import MessageQueue
from umrs.metadata import check_and_build_metadata
from umrs.modality import *
from umrs.json_utils import load_client_cfg_file 
from umrs.pretty_print import pretty_print_user_cfg
from glob import glob
import argparse
from collections import defaultdict
import traceback
import json as _json


# ======================= Helper / API abstractions (new) =======================

def build_bucket_store(user_cfg):
    bucket_store = defaultdict(dict)
    configure_bucket_store(user_cfg, bucket_store)
    return bucket_store

def find_bucket_entry(bucket_store, bucket_alias):
    if not bucket_store[bucket_alias]:
        raise ValueError(f"Unknown bucket alias '{bucket_alias}' in configuration")
    return bucket_store[bucket_alias]

def api_json(status="ok", output_format='json', **payload):
    out = {"status": status}
    out.update(payload)
    if output_format == 'json':
        return _json.dumps(out, ensure_ascii=False)
    lines = []
    for k, v in out.items():
        if isinstance(v, (list, tuple)):
            lines.append(f"{k}=[{', '.join(map(str,v))}]")
        else:
            lines.append(f"{k}={v}")
    return "\n".join(lines)

def parse_metadata_kv(pairs_list):
    meta = {}
    if not pairs_list:
        return meta
    for p in pairs_list:
        if ':' not in p:
            raise ValueError(f"Metadata must be key:value got '{p}'")
        k, v = p.split(':', 1)
        meta[k.strip()] = v.strip()
    return meta

def merge_metadata(base_metadata, override_meta_dict):
    # base_metadata is the structure produced by metadata_dict(sql_conf) -> includes SQL definitions
    # For now we only inject overrides into a flat list under Metadata (matching existing code paths)
    # Existing code path expects metadata_args = { 'Metadata': {...}}
    merged = {k: v for k, v in base_metadata.items()} if base_metadata else {}
    md = merged.get('Metadata', {})
    md.update(override_meta_dict)
    merged['Metadata'] = md
    return merged

def action_ping(control_mq, output_format):
    ready = control_mq.is_server_ready(timeout_ms=2000)
    return api_json(server_ready=ready, output_format=output_format)

def action_list_buckets(bucket_store, username, output_format):
    buckets = []
    for alias, entry in bucket_store.items():
        if entry and entry['username'] == username:
            buckets.append({
                'bucket_alias': alias,
                'modality': entry['modality_name'],
                'path_to_bucket': entry['path_to_bucket']
            })
    return api_json(buckets=buckets, output_format=output_format)

def perform_upload_text(s3, full_bucket_name, text, data_mq, metadata_args):
    s3_key = s3_store_txt(s3, full_bucket_name, text, data_mq, metadata_args)
    return s3_key

def perform_upload_image(s3, full_bucket_name, file_path, data_mq, metadata_args):
    return s3_store_image(s3, full_bucket_name, file_path, data_mq, metadata_args)

def perform_upload_textfile(s3, full_bucket_name, file_path, data_mq, metadata_args):
    return s3_store_text_file(s3, full_bucket_name, file_path, data_mq, metadata_args)

def resolve_bucket(bucket_store, bucket_alias):
    entry = find_bucket_entry(bucket_store, bucket_alias)
    return (entry['path_to_bucket'], entry['username'], entry['xp_name'], entry['modality_name'],
            entry['s3_configuration_file'], entry['metadata'])


def send_files_to_server(username, file_path_list, control_mq, data_mq):
    for file_path in file_path_list:
        data_mq.send_file(file_path)
    success = control_mq.notify_server(username, "FILE", len(file_path_list))
    if success:
        print(f"Files {file_path_list} recevied by server")
    else:
        print(f"Error sending file(s) {file_path_list} to server")
        

def store_bucket(bucket_store, bucket_alias, path_to_bucket, username, xp_name, modality_name, user, s3_configuration_file, metadata, sql_username, db_name):
    if(bucket_store[bucket_alias]):
        sys.exit(f"Error: duplicate bucket_alias {bucket_alias}")
    bucket_store[bucket_alias]= {"path_to_bucket": path_to_bucket,
                                "username": username,
                                "xp_name": xp_name,
                                "modality_name": modality_name,
                                 "s3_configuration_file": s3_configuration_file,
                                 "metadata": metadata,
                                 "sql_username": sql_username,
                                 "db_name": db_name}


def configure_bucket_store(user_cfg, bucket_store):
    fullname = user_cfg["fullname"]
    username = user_cfg["username"]
    s3_configuration_file = user_cfg["s3_configuration_file"]
    for xp in user_cfg["experiences"]:
        sql_username  = xp["sql_username"]
        xp_name = xp["xp_name"]
        db_name = create_db_name(sql_username, xp_name)
        for modality in xp["modalities"]:
            modality_name = modality["modality_name"]
            check_or_quit(modality_name)
            bucket_alias =  modality["s3"]["bucket_alias"]
            path_to_bucket = create_path_to_bucket(xp_name, bucket_alias, username)
            # print(bucket_alias, username, xp_name, modality_name)
            metadata = metadata_dict(modality["sql"])
            store_bucket(bucket_store, bucket_alias, path_to_bucket, username, xp_name, modality_name, user, s3_configuration_file, metadata, sql_username, db_name)



parser = argparse.ArgumentParser(description="UMRS client", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
subparsers = parser.add_subparsers(dest="action", help="API action (optional; legacy mode if omitted)")

# Global / legacy flags
parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose mode")
parser.add_argument('-c', '--conf', default='./conf_umrs.json', help='Config file path')
parser.add_argument("-k", action="append", help="Metadatata key:value", required=False)
parser.add_argument('--no-check-server', action='store_true', help='Do not check if the server is ready. Just copy data in the S3 server if no ready')
parser.add_argument('--server-ip', default='127.0.0.1', help='Server IP for ZeroMQ control channel')
parser.add_argument('--output', choices=['json','text'], default='json', help='Output format for API actions')
parser.add_argument('--auto-add-columns', action='store_true', help='Request server to auto-add missing typed metadata columns')

# Legacy positional arguments (only used when no action)
parser.add_argument("data", type=str, nargs="?", help="input data (legacy mode)", default=None)
parser.add_argument("bucket", type=str, nargs="?", help="target bucket alias (legacy mode)", default=None)

# API subcommands
ping_p = subparsers.add_parser('ping', help='Check server readiness')

ut_p = subparsers.add_parser('upload_text', help='Upload inline text object')
ut_p.add_argument('text', help='Inline text content')
ut_p.add_argument('bucket_alias', help='Target bucket alias')
ut_p.add_argument('username', nargs='?', help='Configured username owning the bucket (defaults to config if omitted)')

ui_p = subparsers.add_parser('upload_image', help='Upload image file')
ui_p.add_argument('file_path')
ui_p.add_argument('bucket_alias')
ui_p.add_argument('username', nargs='?', help='Configured username owning the bucket (defaults to config if omitted)')

utf_p = subparsers.add_parser('upload_textfile', help='Upload text file contents')
utf_p.add_argument('file_path')
utf_p.add_argument('bucket_alias')
utf_p.add_argument('username', nargs='?', help='Configured username owning the bucket (defaults to config if omitted)')

# New actions
lb_p = subparsers.add_parser('list_buckets', help='List bucket aliases for a username')
lb_p.add_argument('username', nargs='?', help='Username to filter buckets (defaults to config if omitted)')

utb_p = subparsers.add_parser('upload_text_batch', help='Upload multiple text objects (batch)')
utb_p.add_argument('bucket_alias')
utb_p.add_argument('username', nargs='?', help='Configured username owning the bucket (defaults to config if omitted)')
utb_grp = utb_p.add_mutually_exclusive_group(required=True)
utb_grp.add_argument('--file', help='File containing one text object per line')
utb_grp.add_argument('--texts', nargs='+', help='Inline text objects')
# Metadata is provided via repeated -k flags like legacy mode

# Embedding maintenance actions
subparsers.add_parser('recompute', help='Recompute all embeddings for all buckets (server-side)')
subparsers.add_parser('update', help='Update embeddings only for missing/mismatched entries (server-side)')
subparsers.add_parser('check', help='Check consistency between S3 objects and DB (server-side)')
subparsers.add_parser('add_columns', help='Auto-add missing typed metadata columns on server-side tables')

args = parser.parse_args()
legacy_mode = (args.action is None)

# Connect to message queue
control_mq = MessageQueue(role = "client", queue_type = 'control', server_ip=args.server_ip)

if not args.no_check_server:
    if not control_mq.is_server_ready(timeout_ms=5000):
        sys.exit("Server connection timeout...\n"
                 "Cannot continue!"
                 )

print (f"Server ready.\nConnected to zeroMQ message queue'")

# Optional one-shot: request server to auto-add missing typed metadata columns
if args.auto_add_columns:
    client_id = os.environ.get('UMRS_CLIENT_ID', 'client')
    try:
        ok = control_mq.notify_server(client_id, 'ADD_COLUMNS', 0)
        if ok:
            print(api_json(action='add_columns', accepted=True, output_format=args.output))
        else:
            print(api_json(status='error', action='add_columns', accepted=False, output_format=args.output))
    except Exception as e:
        print(api_json(status='error', action='add_columns', message=str(e), output_format=args.output))
    if legacy_mode and args.action is None:
        control_mq.close()
        sys.exit(0)

# Path to the configuration file (only required for config-backed actions)
configure_file = args.conf

# Only enforce config file for legacy mode or actions that need bucket/user config
requires_config = legacy_mode or (args.action in {
    'list_buckets', 'upload_text', 'upload_image', 'upload_textfile', 'upload_text_batch'
})
if requires_config and not os.path.exists(configure_file):
    sys.exit(f"Configuration file \"{configure_file}\" not found.")

user = None
bucket_store = None
if requires_config:
    user = load_client_cfg_file(configure_file)
    bucket_store = build_bucket_store(user)
    if args.verbose:
        print("Configuration loaded successfully.")


legacy_mode = (args.action is None)

try:
    if legacy_mode:
        if args.data is None or args.bucket is None:
            print("Misssing data and/or bucket.")
            pretty_print_user_cfg(bucket_store)
            sys.exit(
                "Error: You must provide both the data and the bucket alias.\n"
                "These are required to upload data to the S3 server according to the configuration above."
            )
        if args.verbose:
            pretty_print_user_cfg(bucket_store)
        bucket_alias = args.bucket
        entry = find_bucket_entry(bucket_store, bucket_alias)
        path_to_bucket        = entry['path_to_bucket']
        username              = entry['username']
        xp_name               = entry['xp_name']
        modality              = entry['modality_name']
        s3_configuration_file = entry['s3_configuration_file']
        metadata              = entry['metadata']
        data_mq = control_mq.get_data_queue(username)
        # Removed previous hard-coded test file transfer
    else:
        # API mode: pre-initialize a data queue for the provided username in action args (if any)
        # Some actions require username (upload_*), ping does not.
        if args.action == 'ping':
            print(action_ping(control_mq, args.output))
            control_mq.close()
            sys.exit(0)
        if args.action == 'list_buckets':
            effective_username = args.username or (user.get('username') if isinstance(user, dict) else None)
            if not effective_username:
                print(api_json(status='error', message='No username provided and none found in config', output_format=args.output))
                control_mq.close()
                sys.exit(1)
            print(action_list_buckets(bucket_store, effective_username, args.output))
            control_mq.close()
            sys.exit(0)
        # Embedding maintenance actions don't require bucket resolution or S3 access
        if args.action in ('recompute', 'update', 'check', 'add_columns'):
            # Use configured username as client id for control plane
            # Determine a stable client id for control-plane requests
            client_id = (user.get('username') if isinstance(user, dict) else None) or os.environ.get('UMRS_CLIENT_ID', 'client')
            cmd = {'recompute': 'RECOMPUTE', 'update': 'UPDATE', 'check': 'CHECK', 'add_columns': 'ADD_COLUMNS'}[args.action]
            try:
                ok = control_mq.notify_server(client_id, cmd, 0)
                if ok:
                    print(api_json(action=args.action, accepted=True, output_format=args.output))
                else:
                    print(api_json(status='error', action=args.action, accepted=False, output_format=args.output))
            except Exception as e:
                print(api_json(status='error', action=args.action, message=str(e), output_format=args.output))
            finally:
                control_mq.close()
            sys.exit(0)
        # For upload_* actions we need configuration + bucket alias
        bucket_alias = getattr(args, 'bucket_alias')
        entry = find_bucket_entry(bucket_store, bucket_alias)
        path_to_bucket        = entry['path_to_bucket']
        # If username not provided as arg, default to config
        username_arg = getattr(args, 'username', None)
        username              = username_arg or entry['username']
        xp_name               = entry['xp_name']
        modality              = entry['modality_name']
        s3_configuration_file = entry['s3_configuration_file']
        metadata              = entry['metadata']
        data_mq = control_mq.get_data_queue(username)
except Exception as e:
    print(api_json(status="error", message=str(e), output_format=args.output))
    if args.verbose:
        traceback.print_exc()
    control_mq.close()
    sys.exit(1)

# Allow fallback to default docker s3cfg if user-specific config is missing inside container
expanded_cfg = os.path.expanduser(s3_configuration_file)
if not os.path.exists(expanded_cfg) and os.path.exists("/app/docker/s3cfg"):
    expanded_cfg = "/app/docker/s3cfg"

s3, bucket_location = connect_to_s3(expanded_cfg)

full_bucket_name, success = check_bucket_name(s3, xp_name, bucket_alias, username)

if not success or full_bucket_name != path_to_bucket:
    sys.exit(f"No bucket {bucket_alias} for  modality {modality} and for user {username}")

if args.verbose or (not legacy_mode):
    print(f"Bucket '{bucket_alias}' for  modality '{modality}' and for user '{username}': OK!")

try:
    metadata_args = check_and_build_metadata(metadata, args.k)
except Exception as e:
    print(api_json(status="error", message=f"Metadata error: {e}", output_format=args.output))
    control_mq.close(); data_mq.close(); sys.exit(1)

# In API mode, metadata is provided via repeated -k flags like legacy mode

if args.verbose or (not legacy_mode):
    print(f"Storing data for:\n\tUser: '{username}'\n\tXP: '{xp_name}'\n\tModality: '{modality}'")

server_notify_result = None

def legacy_flow():
    global server_notify_result
    if modality == "text" :
        if os.path.isfile(args.data):
            try:
                with open(args.data,"r") as file:
                    nb_objects = 0
                    for line in file.readlines():
                        line = line.strip()
                        s3_key = s3_store_txt(s3, full_bucket_name, line, data_mq, metadata_args)
                        nb_objects += 1
                        print(f"'{line}' saved as {s3_key} on: {full_bucket_name}")
                    server_notify_result = control_mq.notify_server(username, "OBJECT", nb_objects)
            except UnicodeDecodeError as e:
                print(f"Unicode decoding error: {e}")
                print(f"{args.data} is not a text file. Cannot use modality: '{modality}' for this file.")
        else:
            s3_key = s3_store_txt(s3, full_bucket_name, args.data, data_mq, metadata_args)
            print(f"'{args.data}' saved as {s3_key} on: {full_bucket_name}")
            server_notify_result = control_mq.notify_server(username, "OBJECT", 1)
    elif modality == "image" :
        if os.path.isfile(args.data):
            s3_key = s3_store_image(s3, full_bucket_name, args.data, data_mq, metadata_args)
            if s3_key:
                print(f"'{s3_key}' saved on: {full_bucket_name}")
                server_notify_result = control_mq.notify_server(username, "OBJECT", 1)
        elif os.path.isdir(args.data):
            nb_objects = 0
            for file in glob(os.path.join(args.data, "*")):
                if os.path.isfile(file):
                    s3_key = s3_store_image(s3, full_bucket_name, file, data_mq, metadata_args)
                    if s3_key:
                        nb_objects += 1
                        print(f"'{s3_key}' saved on: {full_bucket_name}")
            server_notify_result = control_mq.notify_server(username, "OBJECT", nb_objects)
        else:
            print(f"Error: do not know what to do with '{args.data}'...")
            print("Provide a path to an image or to a directory of images")
    elif modality == "text file" :
        if os.path.isfile(args.data):
            s3_key = s3_store_text_file(s3, full_bucket_name, args.data, data_mq, metadata_args)
            if s3_key:
                print(f"'{s3_key}' saved on: {full_bucket_name}")
                server_notify_result = control_mq.notify_server(username, "OBJECT", 1)
        elif os.path.isdir(args.data):
            nb_objects = 0
            for file in glob(os.path.join(args.data, "*")):
                if os.path.isfile(file):
                    s3_key = s3_store_text_file(s3, full_bucket_name, file, data_mq, metadata_args)
                    if s3_key:
                        nb_objects += 1
                        print(f"'{s3_key}' saved on: {full_bucket_name}")
            server_notify_result = control_mq.notify_server(username, "OBJECT", nb_objects)
        else:
            print(f"Error: do not know what to do with '{args.data}'...")
            print("Provide a path to a text file or to a directory of text_files")
    else:
        print(f"Error: Unknown modality '{modality}'. Nothing has been sent to the server")

def api_flow():
    # Determine action specifics
    action = args.action
    try:
        if action == 'upload_text':
            if modality != 'text':
                print(api_json(status='error', message=f"Bucket alias '{bucket_alias}' modality mismatch: expected 'text' got '{modality}'", output_format=args.output))
                return
            s3_key = perform_upload_text(s3, full_bucket_name, args.text, data_mq, metadata_args)
            control_mq.notify_server(username, 'OBJECT', 1)
            print(api_json(s3_key=s3_key, bucket=full_bucket_name, modality=modality, output_format=args.output))
        elif action == 'upload_image':
            if modality != 'image':
                print(api_json(status='error', message=f"Bucket alias '{bucket_alias}' modality mismatch: expected 'image' got '{modality}'", output_format=args.output))
                return
            s3_key = perform_upload_image(s3, full_bucket_name, args.file_path, data_mq, metadata_args)
            if s3_key:
                control_mq.notify_server(username, 'OBJECT', 1)
                print(api_json(s3_key=s3_key, bucket=full_bucket_name, modality=modality, output_format=args.output))
            else:
                print(api_json(status='error', message='File not uploaded (invalid image?)', output_format=args.output))
        elif action == 'upload_textfile':
            if modality != 'text file':
                print(api_json(status='error', message=f"Bucket alias '{bucket_alias}' modality mismatch: expected 'text file' got '{modality}'", output_format=args.output))
                return
            s3_key = perform_upload_textfile(s3, full_bucket_name, args.file_path, data_mq, metadata_args)
            if s3_key:
                control_mq.notify_server(username, 'OBJECT', 1)
                print(api_json(s3_key=s3_key, bucket=full_bucket_name, modality=modality, output_format=args.output))
            else:
                print(api_json(status='error', message='File not uploaded (not a text file?)', output_format=args.output))
        elif action == 'upload_text_batch':
            if modality != 'text':
                print(api_json(status='error', message=f"Bucket alias '{bucket_alias}' modality mismatch: expected 'text' got '{modality}'", output_format=args.output))
                return
            texts = []
            if getattr(args, 'file', None):
                if not os.path.isfile(args.file):
                    print(api_json(status='error', message=f"File '{args.file}' not found", output_format=args.output))
                    return
                with open(args.file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line=line.strip()
                        if line:
                            texts.append(line)
            else:
                texts = getattr(args, 'texts', [])
            if not texts:
                print(api_json(status='error', message='No text objects provided', output_format=args.output))
                return
            s3_keys = []
            for t in texts:
                k = perform_upload_text(s3, full_bucket_name, t, data_mq, metadata_args)
                s3_keys.append(k)
            control_mq.notify_server(username, 'OBJECT', len(s3_keys))
            print(api_json(s3_keys=s3_keys, count=len(s3_keys), bucket=full_bucket_name, modality=modality, output_format=args.output))
        else:
            print(api_json(status='error', message=f'Unknown action {action}', output_format=args.output))
    except Exception as e:
        print(api_json(status='error', message=str(e), output_format=args.output))
        if args.verbose:
            traceback.print_exc()

if legacy_mode:
    legacy_flow()
else:
    api_flow()

if legacy_mode:
    if server_notify_result is not None:
        if server_notify_result is True:
            print("Server received data")
        else:
            print("Error: Server did not data")

# close queues before exiting
control_mq.close()
data_mq.close()
