"""
Example user-defined step: image inversion.

Reads images from S3, inverts pixel values (255 - pixel), saves results back.
This is a minimal demonstration — no GPU, no heavy deps.
"""

from umrs.pipeline_sdk import Step, StepContext, StepResult


class InvertImageStep(Step):
    """Invert grayscale or RGB images (negative effect)."""

    name = "invert"

    def setup(self, config: dict) -> None:
        self.output_prefix = config.get("output_prefix", "inverted/")

    def run(self, ctx: StepContext) -> StepResult:
        from PIL import Image
        import io

        output_keys = []
        checksums = []
        metadata_list = []

        for key in ctx.input_keys:
            # Download the image bytes
            raw = ctx.download_bytes(key)

            # Invert
            img = Image.open(io.BytesIO(raw))
            from PIL import ImageOps

            inverted = ImageOps.invert(img.convert("RGB"))

            # Encode back to PNG
            buf = io.BytesIO()
            inverted.save(buf, format="PNG")
            out_key = f"{self.output_prefix}{key}"
            ctx.upload_bytes(out_key, buf.getvalue(), content_type="image/png")

            output_keys.append(out_key)

            # Collect S3 metadata from original
            head = ctx.head_object(key)
            checksums.append(head["ETag"].strip('"'))
            metadata_list.append(head.get("Metadata", {}))

        return StepResult(
            output_keys=output_keys,
            checksums=checksums,
            metadata=metadata_list,
        )
