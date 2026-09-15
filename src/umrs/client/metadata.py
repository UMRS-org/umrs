import sys


def build_metadata_types(metadata):
    metadata_types = {}
    for entry in metadata.values():
        name = entry["name"]
        dtype = entry["type"]
        metadata_types[name] = dtype

    return metadata_types


def check_and_build_metadata(metadata, args):
    if not args:
        return {"Metadata": {}}

    metadata_types = build_metadata_types(metadata)

    metadata_dict = {"Metadata": {}}  # Wrap metadata inside a "Metadata" dictionary

    for arg in args:
        parts = arg.split(":")

        if len(parts) != 2:
            sys.exit(f"Invalid -k metadata format: '{arg}'. Expected 'name:value'.")

        name = parts[0].strip()
        dtype = metadata_types.get(name, None)
        if not dtype:
            sys.exit(
                f"Error: unkonwn metadata name '{name}' in -k argument in the command line\nPossible names are: {', '.join(metadata_types.keys())}"
            )
        value = parts[1].strip()

        metadata_dict["Metadata"][name] = value  # Store metadata inside "Metadata"

    return metadata_dict
