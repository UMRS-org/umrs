#!/usr/bin/env python3

import os
import sys
from umrs.sql import *
from umrs.s3 import *
from umrs.MessageQueue import MessageQueue
from umrs.metadata import check_and_build_metadata
from umrs.modality import *
from umrs.json_utils import load_client_cfg_file
from umrs.pretty_print import pretty_print_user_cfg
from umrs.config import get_default_docker_s3cfg
from glob import glob
import argparse
from collections import defaultdict
import traceback
import json as _json
import time
from umrs.TaskCmdStatus import TaskCmd
from umrs.BucketsCfg import BucketsCfg, ClientBucketEntry


STATUS_CHOICES = TaskCmd.status_cmd()


# ======================= Helper / API abstractions (new) =======================
def configure_buckets(user_cfg, buckets_cfg):
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
            path_to_bucket = make_path_to_bucket_string(xp_name, bucket_alias, username)
            if modality["s3"].get("need_output_bucket", False):
                output_path_to_bucket = make_path_to_bucket_string(xp_name, bucket_alias, username, output=True)
            else:
                output_path_to_bucket = None
            # print(bucket_alias, username, xp_name, modality_name)
            metadata = metadata_dict(modality["sql"])
            entry = ClientBucketEntry(   
                path_to_bucket,
                username,
                xp_name,
                modality_name,
                s3_configuration_file,
                metadata,
                sql_username,
                db_name,
                output_path_to_bucket
            )
            buckets_cfg.store(bucket_alias, entry)

def build_buckets_cfg(user_cfg):
    buckets_cfg = BucketsCfg()
    configure_buckets(user_cfg, buckets_cfg)
    return buckets_cfg

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



def format_status_text(response_payload, default_key):
    if response_payload.get('status') != 'ok':
        message = response_payload.get('message', 'status unavailable')
        return api_json(status='error', message=message, output_format='text')

    client_label = response_payload.get('client')
    lines = []

    def append_entry(name, entry):
        header_parts = []
        if client_label:
            header_parts.append(client_label)
        if name:
            header_parts.append(name)
        generated_at = entry.get('generated_at')
        if header_parts or generated_at:
            header = ' - '.join(header_parts)
            if generated_at:
                header = f"{header} (generated {generated_at})" if header else f"generated {generated_at}"
            if header:
                lines.append(header)
        summary_body = entry.get('summary')
        if summary_body:
            lines.append(summary_body)
        else:
            lines.append("(no summary)")

    if 'summary' in response_payload:
        append_entry(default_key, response_payload['summary'])
    elif 'summaries' in response_payload:
        summaries = response_payload['summaries'] or {}
        for key in sorted(summaries.keys()):
            append_entry(key, summaries[key])
            lines.append("")
        if lines and lines[-1] == "":
            lines.pop()
    else:
        return api_json(status='error', message='Unexpected status payload', output_format='text')

    return "\n".join(lines)

def action_ping(control_mq, output_format):
    ready = control_mq.is_server_ready(timeout_ms=2000)
    return api_json(server_ready=ready, output_format=output_format)

def action_list_buckets(bucket_cfg, username, output_format):
    buckets = []
    for alias, entry in bucket_cfg.items():
        if entry.username == username :
            buckets.append({
                'bucket_alias': alias,
                'modality': entry.modality_name,
                'path_to_bucket': entry.path_to_bucket
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
    success = control_mq.notify_server(username, TaskCmd.FILE, len(file_path_list))
    if success:
        print(f"Files {file_path_list} received by server")
    else:
        print(f"Error sending file(s) {file_path_list} to server")





parser = argparse.ArgumentParser(description="UMRS client", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
subparsers = parser.add_subparsers(dest="action", help="API action", required=True)

# Global / legacy flags
parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose mode")
parser.add_argument('-c', '--conf', default='./conf_umrs.json', help='Config file path')
parser.add_argument("-k", action="append", help="Metadatata key:value", required=False)
parser.add_argument('--no-check-server', action='store_true', help='Do not check if the server is ready. Just copy data in the S3 server if no ready')
parser.add_argument('--server-ip', default='127.0.0.1', help='Server IP for ZeroMQ control channel')
parser.add_argument('--output', choices=['json','text'], default='json', help='Output format for API actions')
# Removed deprecated auto-add-columns behavior; metadata must be declared in server config

# API subcommands
ping_p = subparsers.add_parser('ping', help='Check server readiness')

ui_p = subparsers.add_parser('upload_file', help="Upload file to a bucket (ommit empty lines in case of 'text' modality)")
ui_p.add_argument('file_path')
ui_p.add_argument(
    'bucket_alias',
    help=(
        "Alias of a bucket as defined in the client configuration file.\n"
        "Use the list_bucket command to show available bucket aliases for a given user.\n"
        "You may append a subdirectory using '/' as in a typical Unix path."
    )
)


ui_p.add_argument('username', nargs='?', help='Configured username owning the bucket (defaults to config if omitted)')

utf_p = subparsers.add_parser('upload_dir', help='Upload all files of a directory to a bucket')
utf_p.add_argument('path')
utf_p.add_argument(
    'bucket_alias',
    help=(
        "Alias of a bucket as defined in the client configuration file.\n"
        "Use the list_bucket command to show available bucket aliases for a given user.\n"
        "You may append a subdirectory using '/' as in a typical Unix path."
    )
)
utf_p.add_argument('username', nargs='?', help='Configured username owning the bucket (defaults to config if omitted)')

dr_p = subparsers.add_parser('download_results', help="Download file results computed by the server")
dr_p.add_argument(
    'bucket_alias',
    help=(
        "Alias of a bucket as defined in the client configuration file.\n"
        "Use the list_bucket command to show available bucket aliases for a given user.\n"
        "You may append a subdirectory using '/' as in a typical Unix path."
    )
)
dr_p.add_argument(
    'local_path',
    help=(
        "local Unix path where the results will be stored.\n"
        "New files will erase local files."
    )
)

dr_p.add_argument(
    '--force-overwrite',
    action='store_true',
    default=False,
    help="Force overwriting existing local files even if checksums mismatch."
)
dr_p.add_argument('username', nargs='?', help='Configured username owning the bucket (defaults to config if omitted)')

# New actions
lb_p = subparsers.add_parser('list_buckets', help='List bucket aliases for a username')
lb_p.add_argument('username', nargs='?', help='Username to filter buckets (defaults to config if omitted)')

status_p = subparsers.add_parser('status', help='Fetch the latest server summary for maintenance commands')
status_p.add_argument('target', choices=STATUS_CHOICES, metavar='key', nargs='?', default='all', help=f'Summary to fetch (options: {", ".join(STATUS_CHOICES)}; defaults to all)')#help='Summary to fetch (defaults to all available summaries)')
status_p.add_argument('--username', help='User whose summary should be retrieved (defaults to config if omitted)')

# Embedding maintenance actions
subparsers.add_parser('recompute', help='Recompute all embeddings for all buckets (server-side)')
subparsers.add_parser('update', help='Update embeddings only for missing/mismatched entries (server-side)')
subparsers.add_parser('check', help='Check consistency between S3 objects and DB (server-side)')
rl_p = subparsers.add_parser('reload', help='Reload server configuration (users, buckets, models) without restart; if -c points to a user JSON, it will be uploaded and persisted on the server before reloading')


args = parser.parse_args()


# Connect to message queue
control_mq = MessageQueue(role = "client", queue_type = 'control', server_ip=args.server_ip)

if not args.no_check_server:
    if not control_mq.is_server_ready(timeout_ms=5000):
        sys.exit("Server connection timeout...\n"
                 "Cannot continue!"
                 )

print (f"Server ready.\nConnected to zeroMQ message queue'")

# Deprecated 'auto_add_columns' flag removed

# Path to the configuration file (only required for config-backed actions)
configure_file = args.conf

# Only enforce config file for legacy mode or actions that need bucket/user config
status_needs_config = args.action == 'status' and not getattr(args, 'username', None)
requires_config =  args.action in {
    'list_buckets', 'upload_file', 'upload_dir', "download_results", 
    "recompute", "update", "reload"
} or status_needs_config

if requires_config and not os.path.exists(configure_file):
    sys.exit(f"Configuration file \"{configure_file}\" not found.")

user = None
buckets_cfg = None
if requires_config:
    user = load_client_cfg_file(configure_file)
    buckets_cfg = build_buckets_cfg(user)
    if args.verbose:
        print("Configuration loaded successfully.")



try:
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
        print(action_list_buckets(buckets_cfg, effective_username, args.output))
        control_mq.close()
        sys.exit(0)
    if args.action == 'status':
        effective_username = args.username
        if not effective_username and isinstance(user, dict):
            effective_username = user.get('username')
        if not effective_username:
            print(api_json(status='error', message='No username provided and none found in config', output_format=args.output))
            control_mq.close()
            sys.exit(1)
        try:
            raw_response = control_mq.request_status(effective_username, args.target)
            if args.output == 'json':
                print(raw_response)
            else:
                try:
                    parsed = _json.loads(raw_response)
                except Exception as exc:
                    print(api_json(status='error', message=f'Invalid status payload: {exc}', output_format=args.output))
                    control_mq.close()
                    sys.exit(1)
                print(format_status_text(parsed, args.target))
        except Exception as e:
            print(api_json(status='error', action=args.action, message=str(e), output_format=args.output))
        finally:
            control_mq.close()
        sys.exit(0)
    # Embedding maintenance actions don't require bucket resolution or S3 access
    if args.action in ('recompute', 'update', 'check', 'reload'):
        # Use configured username as client id for control plane
        # Determine a stable client id for control-plane requests
        client_id = (user.get('username') if isinstance(user, dict) else None) or os.environ.get('UMRS_CLIENT_ID', '_')
        cmd = {'recompute': TaskCmd.RECOMPUTE, 'update': TaskCmd.UPDATE, 'check': TaskCmd.CHECK, 'reload': TaskCmd.RELOAD}[args.action]
        try:
            # Special handling for reload with a user-config file: push file first, then RELOAD
            if args.action == 'reload':
                # If -c points to an existing user.json, upload it first so the server persists and reloads it
                cfg_path = args.conf if getattr(args, 'conf', None) else None
                if cfg_path and os.path.isfile(cfg_path):
                    try:
                        with open(cfg_path, 'r', encoding='utf-8') as f:
                            cfg_json = _json.load(f)
                        upload_user = cfg_json.get('username') or client_id
                    except Exception:
                        upload_user = client_id
                    data_mq = control_mq.get_data_queue(upload_user)
                    try:
                        data_mq.send_file(cfg_path)
                        control_mq.notify_server(upload_user, TaskCmd.FILE, 1)
                    finally:
                        try:
                            data_mq.close()
                        except Exception:
                            pass
            # Now send the control-plane command
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
    bucket_alias_arg = getattr(args, 'bucket_alias')
    bucket_alias, subdir = decompose_bucket_alias_arg(bucket_alias_arg)
    
    selected_fields = ["username", "xp_name", "modality_name", "s3_configuration_file", "metadata"]
    if args.action == "download_results":
        selected_fields.append("output_path_to_bucket")
    else : 
        selected_fields.append("path_to_bucket")
    username, xp_name, modality, s3_configuration_file, metadata, path_to_bucket = buckets_cfg.get(bucket_alias, filter=selected_fields)

    # If username is provided as arg change it otherwise stay default
    username_arg = getattr(args, 'username', None)
    username = username_arg or username

    data_mq = control_mq.get_data_queue(username)
except Exception as e:
    print(api_json(status="error", message=str(e), output_format=args.output))
    if args.verbose:
        traceback.print_exc()
    control_mq.close()
    sys.exit(1)

# Allow fallback to default docker s3cfg if user-specific config is missing inside container
expanded_cfg = os.path.expanduser(s3_configuration_file)
default_docker_s3cfg = get_default_docker_s3cfg()
if not os.path.exists(expanded_cfg) and os.path.exists(default_docker_s3cfg):
    expanded_cfg = default_docker_s3cfg

s3, bucket_location = connect_to_s3(expanded_cfg)

if args.action == "download_results":
    full_bucket_name, success = check_bucket_name(s3, xp_name, bucket_alias, username, output=True, indent=False)
    if not success or full_bucket_name != path_to_bucket:
            sys.exit(f"Error:\tNo output bucket for {bucket_alias} for  modality {modality} and for user {username}\n"
                     f"\tCheck that this modality has 'need_output_bucket' set to 'true'")
else :
    full_bucket_name, success = check_bucket_name(s3, xp_name, bucket_alias, username, indent=False)
    if not success or full_bucket_name != path_to_bucket:
        sys.exit(f"Error:\tNo bucket {bucket_alias} for modality {modality} and for user {username}")

if args.verbose :
    print(f"Bucket '{bucket_alias}' for modality '{modality}' and for user '{username}': OK!")

try:
    metadata_args = check_and_build_metadata(metadata, args.k)
except Exception as e:
    print(api_json(status="error", message=f"Metadata error: {e}", output_format=args.output))
    control_mq.close(); data_mq.close(); sys.exit(1)

# In API mode, metadata is provided via repeated -k flags like legacy mode

if args.verbose :
   action = "Retreiving" if args.action == "download_results" else "Storing"
   print(f"{action} data for:\n\tUser: '{username}'\n\tXP: '{xp_name}'\n\tModality: '{modality}'")

server_notify_result = None


def process_file(s3, full_bucket_name, subdir, file_path, modality, data_mq, metadata_args, username, notify_server=True):
    if not os.path.isfile(file_path):
        print(f"Error: file {file_path} is not a valid.")
        return None, 0

    print(f"Processing file '{file_path}' for modality '{modality}'")

    if modality in ["image" , "tomography_embedding", "tomography_reconstruction", "tomography_mae_reconstruction", "tomography_features_extraction"]: # send an image
        if os.path.isfile(file_path):
            s3_key = s3_store_image(s3, full_bucket_name, subdir, file_path, data_mq, metadata_args, check = (modality!="tomography"))
            if s3_key and notify_server:
                control_mq.notify_server(username, TaskCmd.PROCESS_OBJECTS, 1)
            return s3_key, 1
    elif modality == "text file" : # send a text file by chunking it
            s3_key = s3_store_text_file(s3, full_bucket_name, subdir, file_path, data_mq, metadata_args)
            if s3_key and notify_server:
                control_mq.notify_server(username, TaskCmd.PROCESS_OBJECTS, 1)
            return s3_key, 1
    elif modality == "text": # consider each line of the file as a string to be process
        try:
            with open(file_path,"r") as file:
                nb_objects = 0
                s3_key_list = []
                for line in file.readlines():
                    line = line.strip()
                    if line != "" : # skip empty lines as it brings nothing and as it slows down the process 
                        s3_key = s3_store_txt(s3, full_bucket_name, subdir, line, data_mq, metadata_args)
                        s3_key_list.append(s3_key)
                        nb_objects += 1
                if notify_server and nb_objects > 0:
                    control_mq.notify_server(username, TaskCmd.PROCESS_OBJECTS, nb_objects)
                return s3_key_list, nb_objects
        except UnicodeDecodeError as e:
            print(f"Unicode decoding error: {e}")
            print(f"{args.data} is not a text file. Cannot use modality: '{modality}' for this file.")
    else:
        print(f"Error: Unknown modality '{modality}'. Nothing has been sent to the server")

    return None, 0

def process_dir(path, s3, full_bucket_name, subdir, modality, data_mq, metadata_args, username, control_mq):
    if not os.path.isdir(path):
        print(f"Error: {path} is not a valid directory.")
        return [], []

    s3_key_list = []
    file_list = []
    nb_objects= 0
    for file_path in glob(os.path.join(path, "*")):
        if os.path.isfile(file_path):
            s3_key, nb_sent = process_file(s3, full_bucket_name, subdir, file_path, modality, data_mq, metadata_args, username, notify_server=False)
            if s3_key:
                if isinstance(s3_key, list): # files of text are processed line by line (each leading to a file stored on the s3 server)
                    s3_key_list.extend(s3_key)  # add each element individually if s3_key is a list
                    assert(len(s3_key)==nb_sent)
                    file_list.extend([file_path] * nb_sent) #ensure that file_list is of same size as s3_key_list
                else:
                    s3_key_list.append(s3_key)
                    file_list.append(file_path)
                nb_objects += nb_sent

    if nb_objects>0:
        control_mq.notify_server(username, TaskCmd.PROCESS_OBJECTS, nb_objects)
    return s3_key_list, file_list

def download_results(args, full_output_bucket_name, modality, subdir):
    effective_username = args.username
    verbose = args.verbose
    if not effective_username and isinstance(user, dict):
        effective_username = user.get('username')
    if not effective_username:
        response = {
            "status": "error",
            "message": f"Cannot find effective username for {args.username}",
            "bucket": full_output_bucket_name,
            "subdir": subdir,
            "modality": modality,
            "output_format": args.output,
        }
        return False, response
        

    # check if the target path exists
    if not os.path.exists(args.local_path):
        response = {
            "status": "error",
            "message": f"Local path {args.local_path} does not exists",
            "bucket": full_output_bucket_name,
            "subdir": subdir,
            "modality": modality,
            "output_format": args.output,
        }
        return False, response
    
    done = False
    print_only_new = False
    files_received = set()
    files_skipped = set()
    timeout = 1

    # check the output bucket until the server has finish processing the input bucket for thi username
    # 
    while not done: # loop until we are sure the server has done with this bucket
        #get the list of objects on s3 server
        objects_list, _, checksums = list_s3_bucket(s3, full_output_bucket_name, prefix=subdir)
        #down load these object talink into account if we force overwrite or not
        downloaded_files_set, skipped_files_set = download_objects(s3, objects_list, checksums, args.local_path, files_received,
                            force_overwrite=args.force_overwrite, verbose=args.verbose, print_only_new=print_only_new)
        files_received.update(downloaded_files_set)
        files_skipped.update(skipped_files_set)
        # for now print only new dowloaded files
        print_only_new = True
        # now we need to check if the server is still computing things for this bucket
        # ask the server
        raw_response = control_mq.request_status(effective_username, TaskCmd.PROCESS_OBJECTS)
        print(f"{raw_response=}")

        if raw_response is None:
            break;        
        
        # raw response is a string of a serialized json. transform it into json dict
        parsed = _json.loads(raw_response)

        # here we treat the case where some tasks did not finish correctely 
        # and warn the user it should end it again.
        status = parsed.get("status")
        message = parsed.get("message")
        uncompleted_key_list = parsed.get("uncompleted_key_list", None)
        if uncompleted_key_list is not None: 
            print("WARNING")            
            # Transforme each string in list
            nested_lists = [_json.loads(sublist) for sublist in uncompleted_key_list]
            # flatten and suppress duplicates
            unique_keys = list(set(item for sublist in nested_lists for item in sublist))
            for key in sorted(unique_keys):
                bucket = key.split('/')[2]
                if bucket in full_output_bucket_name:
                    key = key.split('/')[3]
                    bucket_alias = bucket.split('.')[-1]
                    print(f"Warning: key {key} in bucket alias {bucket_alias}. Did not finsihed correctely. You should ask the sever to recompute it with update command")

        # if there is no entry, we are done
        if  message == "No status entries" or message == "No status for 'process_objects'":
            done=True
        elif status == 'ok' : 
            #if there are entries, check the one that correspond to this bucket
            # first extract data
            # from pprint import pprint
            summary = parsed.get("summary", None)
            if summary is not None:
                data = summary["data"]
                # pprint(data, width=100)
                # Check whether all tasks related to this bucket are finished.
                # If there exists at least one task whose status is not "Done"
                # and whose work bucket matches the requested bucket prefix 
                # and are of processing S3 objectstype ,
                # then the computation is still running.
                done = not any(
                    entry["status"] != "Done"
                    and any(
                        full_output_bucket_name.startswith(w["bucket"]) # and w["work_type"] == "process_objects"
                        for w in entry["work"]
                    )
                    for entry in data
                )
        else:
            print(f"Received {parsed=}")
            print(f"Not implemented. Warn developpers if seen this message")
        if not done : 
            if verbose:
                print("Waiting for update")
            res = control_mq.wait_for_new_data(effective_username, full_output_bucket_name, timeout=timeout)
            if res == "TIMEOUT":
                timeout = min(timeout*2, 64)
            elif res == "UPDATE":
                timeout= 1
            else : 
                if verbose:
                    print(f"Error: receiving '{res}' from server while waiting to be notified for new data")
    # print("Done. Final check...")    
    # Do it one more time in case the server finished after the last check 
    # get the list of objects on s3 server
    objects_list, total_size, checksums = list_s3_bucket(s3, full_output_bucket_name, prefix=subdir)
    #down load these object talink into account if we force overwrite or not
    downloaded_files_set, skipped_files_set = download_objects(s3, objects_list, checksums, args.local_path, files_received, 
                        force_overwrite=args.force_overwrite, print_only_new=print_only_new)
    
    files_received.update(downloaded_files_set)
    files_skipped.update(skipped_files_set)
   
    response = {
            "status": "ok",
            "message": f"Files downloaded with overwrite={args.force_overwrite}",
            "bucket": full_output_bucket_name,
            "local_path": args.local_path, 
            "subdir": subdir,
            "modality": modality,
            "output_format": args.output,
            "files_received": f"{sorted(files_received)}", 
            "files_skipped": f"{sorted(files_skipped)}", 
        }
    return True, response

# Determine action specifics
action = args.action
try:
    if action == "upload_file":
        s3_key, nb_sent = process_file(s3, full_bucket_name, subdir, args.file_path, modality, data_mq, metadata_args, username, notify_server=True)
        if s3_key:
            print(api_json(s3_key=s3_key, bucket=full_bucket_name, subdir=subdir, modality=modality, output_format=args.output))
        else:
            print(api_json(status='error', message=f"Cannot process file: {args.file_path}", bucket=full_bucket_name, subdir=subdir, modality=modality, output_format=args.output))

    elif action == "upload_dir":
        s3_key_list, file_list =  process_dir(args.path, s3, full_bucket_name, subdir, modality, data_mq, metadata_args, username, control_mq)
        if len(s3_key_list)>0:
            print(api_json(s3_key_list=s3_key_list, file_list=file_list, bucket=full_bucket_name, subdir=subdir, modality=modality, output_format=args.output))
        else:
            print(api_json(status='error', message=f"Cannot process directory: {args.path}", bucket=full_bucket_name, subdir=subdir, modality=modality, output_format=args.output))
    elif action == "download_results":
        # here full_bucket_name is the output bucket
        ok, response = download_results(args, full_bucket_name, modality, subdir)
        print(api_json(**response))
        
                            
except Exception as e:
    print(api_json(status='error', message=str(e), output_format=args.output))
    if args.verbose:
        traceback.print_exc()


# close queues before exiting
control_mq.close()
data_mq.close()
