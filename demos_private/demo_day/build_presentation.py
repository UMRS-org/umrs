#!/usr/bin/env python3
"""Generate UMRS Demo Day PowerPoint presentation."""
import os, sys
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from datetime import datetime

# UMRS Brand Colors
UMRS_NAVY = RGBColor(0x00, 0x2B, 0x5C)      # Primary dark blue
UMRS_TEAL = RGBColor(0x00, 0x7A, 0x87)      # Teal accent
UMRS_LIGHT_BLUE = RGBColor(0x00, 0x96, 0xD6)  # Light blue
UMRS_WHITE = RGBColor(0xFF, 0xFF, 0xFF)
UMRS_GRAY = RGBColor(0x66, 0x66, 0x66)
UMRS_LIGHT_GRAY = RGBColor(0xF0, 0xF0, 0xF0)
UMRS_GREEN = RGBColor(0x27, 0xAE, 0x60)
UMRS_ORANGE = RGBColor(0xE6, 0x7E, 0x22)
UMRS_RED = RGBColor(0xE7, 0x4C, 0x3C)

OUTPUT_DIR = Path("/home/mnagaso/workspace/RCCS/ai4s-platform/umrs/demos_private/demo_day")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def set_slide_bg(slide, color):
    """Set slide background color."""
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color

def add_textbox(slide, left, top, width, height, text, font_size=18,
                font_color=UMRS_WHITE, bold=False, alignment=PP_ALIGN.LEFT,
                font_name="Calibri"):
    """Add a text box to the slide."""
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(font_size)
    p.font.color.rgb = font_color
    p.font.bold = bold
    p.font.name = font_name
    p.alignment = alignment
    return txBox

def add_shape_with_text(slide, left, top, width, height, text,
                        fill_color=UMRS_NAVY, font_size=14, font_color=UMRS_WHITE):
    """Add a colored shape with text."""
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_color
    shape.line.fill.background()
    tf = shape.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(font_size)
    p.font.color.rgb = font_color
    p.font.bold = True
    p.alignment = PP_ALIGN.CENTER
    return shape

def add_bullet_list(slide, left, top, width, height, items, font_size=16,
                    font_color=UMRS_WHITE, bullet_char="•"):
    """Add a bulleted list."""
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.text = f"{bullet_char} {item}"
        p.font.size = Pt(font_size)
        p.font.color.rgb = font_color
        p.font.name = "Calibri"
        p.space_after = Pt(6)
    return txBox

def add_header_bar(slide, text):
    """Add a header bar at the top."""
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE,
                                  Inches(0), Inches(0), Inches(13.333), Inches(1.2))
    bar.fill.solid()
    bar.fill.fore_color.rgb = UMRS_NAVY
    bar.line.fill.background()
    
    add_textbox(slide, Inches(0.5), Inches(0.2), Inches(12), Inches(0.8),
                text, font_size=28, font_color=UMRS_WHITE, bold=True)

def add_footer(slide, page_num, total_pages):
    """Add a footer with page number."""
    footer = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE,
                                     Inches(0), Inches(7.4), Inches(13.333), Inches(0.3))
    footer.fill.solid()
    footer.fill.fore_color.rgb = UMRS_TEAL
    footer.line.fill.background()
    
    add_textbox(slide, Inches(0.5), Inches(7.4), Inches(6), Inches(0.3),
                "UMRS Demo Day • Confidential", font_size=10,
                font_color=UMRS_WHITE, alignment=PP_ALIGN.LEFT)
    
    add_textbox(slide, Inches(9), Inches(7.4), Inches(4), Inches(0.3),
                f"Slide {page_num}/{total_pages}", font_size=10,
                font_color=UMRS_WHITE, alignment=PP_ALIGN.RIGHT)

print("Generating UMRS Demo Day PowerPoint...")
print("")

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)

# ========== SLIDE 1: Title Slide ==========
print("Slide 1/10: Title Slide")
slide = prs.slides.add_slide(prs.slide_layouts[6])  # Blank layout
set_slide_bg(slide, UMRS_NAVY)

# Big title
add_textbox(slide, Inches(1), Inches(1.5), Inches(11), Inches(1.5),
            "UMRS", font_size=72, font_color=UMRS_WHITE, bold=True,
            alignment=PP_ALIGN.CENTER)

add_textbox(slide, Inches(1), Inches(2.8), Inches(11), Inches(1),
            "Unified Multimodal Retrieval System", font_size=36,
            font_color=UMRS_TEAL, bold=False, alignment=PP_ALIGN.CENTER)

# Subtitle
add_textbox(slide, Inches(2), Inches(4.2), Inches(9), Inches(0.8),
            "Demo Day • Production Platform Walkthrough", font_size=24,
            font_color=UMRS_LIGHT_BLUE, alignment=PP_ALIGN.CENTER)

# Date
add_textbox(slide, Inches(2), Inches(5.2), Inches(9), Inches(0.5),
            datetime.now().strftime("%B %d, %Y"), font_size=18,
            font_color=UMRS_WHITE, alignment=PP_ALIGN.CENTER)

# Author
add_textbox(slide, Inches(2), Inches(5.8), Inches(9), Inches(0.5),
            "AI4S Platform Team", font_size=16,
            font_color=UMRS_WHITE, alignment=PP_ALIGN.CENTER)

add_footer(slide, 1, 10)

# ========== SLIDE 2: What is UMRS? ==========
print("Slide 2/10: What is UMRS?")
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(slide, UMRS_WHITE)
add_header_bar(slide, "What is UMRS?")

add_textbox(slide, Inches(0.5), Inches(1.5), Inches(5), Inches(3),
            "UMRS is the production platform for:\n\n"
            "• Multimodal data processing\n"
            "  (text, vision, embeddings)\n\n"
            "• AI-powered inference pipelines\n"
            "  (LLMs, vision models, RAG)\n\n"
            "• HPC-integrated GPU acceleration\n"
            "  (Slurm + containers)\n\n"
            "• Secure, audited operations\n"
            "  (multi-tenant, RBAC)",
            font_size=18, font_color=UMRS_NAVY)

# Key stats box
stats_shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                      Inches(8), Inches(1.5), Inches(4.5), Inches(3))
stats_shape.fill.solid()
stats_shape.fill.fore_color.rgb = UMRS_NAVY
stats_shape.line.fill.background()

stats_text = (
    "Production Stats\n\n"
    "✓ 35+ system models\n"
    "✓ 5 bucket aliases\n"
    "✓ RAG service online\n"
    "✓ Workflow engine\n"
    "✓ Complete audit trail"
)
add_textbox(slide, Inches(8.2), Inches(1.7), Inches(4), Inches(2.5),
            stats_text, font_size=14, font_color=UMRS_WHITE)

add_footer(slide, 2, 10)

# ========== SLIDE 3: Architecture ==========
print("Slide 3/10: Architecture")
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(slide, UMRS_WHITE)
add_header_bar(slide, "Platform Architecture")

# Client Layer
add_shape_with_text(slide, Inches(1), Inches(2), Inches(3), Inches(0.8),
                    "Client Layer", fill_color=UMRS_TEAL, font_size=14)

# UMRS Server
add_shape_with_text(slide, Inches(5), Inches(2), Inches(3), Inches(0.8),
                    "UMRS Server", fill_color=UMRS_NAVY, font_size=14)

# Storage Layer
add_shape_with_text(slide, Inches(9), Inches(2), Inches(3), Inches(0.8),
                    "Storage", fill_color=UMRS_LIGHT_BLUE, font_size=14)

# GPU Compute
add_shape_with_text(slide, Inches(5), Inches(4.5), Inches(3), Inches(0.8),
                    "GPU Compute (Slurm)", fill_color=UMRS_ORANGE, font_size=14)

# Arrows (represented as text)
add_textbox(slide, Inches(4), Inches(2.2), Inches(1.2), Inches(0.4),
            "→", font_size=36, font_color=UMRS_TEAL, alignment=PP_ALIGN.CENTER)

add_textbox(slide, Inches(8.2), Inches(2.2), Inches(1), Inches(0.4),
            "→", font_size=36, font_color=UMRS_NAVY, alignment=PP_ALIGN.CENTER)

# Description below
add_textbox(slide, Inches(0.5), Inches(5.8), Inches(12), Inches(1.5),
            "Client: umrs-client CLI, r4 SDK  |  Server: Job orchestration, container lifecycle  |  "
            "Storage: S3/MinIO, PostgreSQL/pgvector  |  Compute: GPU via Slurm",
            font_size=14, font_color=UMRS_GRAY)

add_footer(slide, 3, 10)

# ========== SLIDE 4: Demo 1 — Text Embeddings ==========
print("Slide 4/10: Demo 1 — Text Embeddings & RAG")
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(slide, UMRS_WHITE)
add_header_bar(slide, "Demo 1: Text Embeddings & RAG")

steps = [
    "Create sample text files with scientific content",
    "Upload to bucket: umrs-client data dir-upload /path/to/texts mnagaso3-text",
    "UMRS processes files and generates embeddings",
    "Download results: umrs-client data download mnagaso3-text /results",
    "RAG pipeline enables semantic search over uploaded documents"
]
add_bullet_list(slide, Inches(0.8), Inches(1.5), Inches(6), Inches(5),
                steps, font_size=16, font_color=UMRS_NAVY)

# Command box
cmd_box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                  Inches(8), Inches(1.5), Inches(4.5), Inches(5))
cmd_box.fill.solid()
cmd_box.fill.fore_color.rgb = RGBColor(0x1E, 0x1E, 0x1E)
cmd_box.line.fill.background()

cmd_text = (
    "$ mkdir -p /tmp/rag-texts\n"
    "$ echo 'RAG enables...'\n"
    "  > /tmp/rag-texts/doc.txt\n"
    "\n"
    "$ umrs-client data\n"
    "  dir-upload /tmp/rag-texts\n"
    "  mnagaso3-text\n"
    "\n"
    "Processing file 'doc.txt'...\n"
    "status=ok\n"
    "\n"
    "$ umrs-client data\n"
    "  download mnagaso3-text\n"
    "  /results\n"
    "\n"
    "✓ Downloaded 2 files"
)
add_textbox(slide, Inches(8.2), Inches(1.7), Inches(4), Inches(4.5),
            cmd_text, font_size=11, font_color=RGBColor(0x00, 0xFF, 0x00),
            font_name="Consolas")

add_footer(slide, 4, 10)

# ========== SLIDE 5: Demo 2 — SAM3 Segmentation ==========
print("Slide 5/10: Demo 2 — SAM3.1 Image Segmentation")
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(slide, UMRS_WHITE)
add_header_bar(slide, "Demo 2: SAM3.1 Image Segmentation")

steps = [
    "Upload images to segmentation bucket",
    "Server dispatches images to GPU node via Slurm",
    "SAM3 model generates bounding boxes & masks",
    "Download segmentation results with overlays",
    "Results include: masks, scores, metadata JSON"
]
add_bullet_list(slide, Inches(0.8), Inches(1.5), Inches(6), Inches(5),
                steps, font_size=16, font_color=UMRS_NAVY)

cmd_box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                  Inches(8), Inches(1.5), Inches(4.5), Inches(5))
cmd_box.fill.solid()
cmd_box.fill.fore_color.rgb = RGBColor(0x1E, 0x1E, 0x1E)
cmd_box.line.fill.background()

cmd_text = (
    "$ umrs-client data\n"
    "  dir-upload /images\n"
    "  demo-defect\n"
    "\n"
    "Processing Cat_1.jpg...\n"
    "Processing Dog_1.jpg...\n"
    "status=ok\n"
    "\n"
    "$ umrs-client data\n"
    "  download demo-defect\n"
    "  /segmentation\n"
    "\n"
    "✓ Downloaded 5 files:\n"
    "  • overlay_Cat_1.png\n"
    "  • overlay_Dog_1.png\n"
    "  • masks/*.json\n"
    "  • metadata/*.json"
)
add_textbox(slide, Inches(8.2), Inches(1.7), Inches(4), Inches(4.5),
            cmd_text, font_size=11, font_color=RGBColor(0x00, 0xFF, 0x00),
            font_name="Consolas")

add_footer(slide, 5, 10)

# ========== SLIDE 6: Demo 3 — HMAE Inference ==========
print("Slide 6/10: Demo 3 — HMAE Inference")
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(slide, UMRS_WHITE)
add_header_bar(slide, "Demo 3: HMAE Inference")

steps = [
    "HMAE reconstructs volume slices from projections",
    "Requires GPU node with HMAE model checkpoint",
    "Upload volume slices to HMAE bucket",
    "HMAE processes images with reconstruction task",
    "Download reconstructed images & metadata"
]
add_bullet_list(slide, Inches(0.8), Inches(1.5), Inches(6), Inches(5),
                steps, font_size=16, font_color=UMRS_NAVY)

cmd_box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                  Inches(8), Inches(1.5), Inches(4.5), Inches(5))
cmd_box.fill.solid()
cmd_box.fill.fore_color.rgb = RGBColor(0x1E, 0x1E, 0x1E)
cmd_box.line.fill.background()

cmd_text = (
    "$ umrs-client data\n"
    "  dir-upload /slices\n"
    "  demo-hmae\n"
    "\n"
    "Processing slice_001.png...\n"
    "  → Submitting to HMAE service\n"
    "Processing slice_002.png...\n"
    "  → Submitting to HMAE service\n"
    "status=ok\n"
    "\n"
    "$ umrs-client data\n"
    "  download demo-hmae\n"
    "  /reconstructions\n"
    "\n"
    "✓ Downloaded 20 files:\n"
    "  • reconstructed/*.png\n"
    "  • metadata/*.json"
)
add_textbox(slide, Inches(8.2), Inches(1.7), Inches(4), Inches(4.5),
            cmd_text, font_size=11, font_color=RGBColor(0x00, 0xFF, 0x00),
            font_name="Consolas")

add_footer(slide, 6, 10)

# ========== SLIDE 7: Advanced Features ==========
print("Slide 7/10: Advanced Features")
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(slide, UMRS_WHITE)
add_header_bar(slide, "Advanced Features")

# Feature boxes
features = [
    ("Workflow Engine", "Multi-step pipelines with\nDAG execution & GPU support"),
    ("RAG Pipeline", "Semantic search with\nPostgreSQL + pgvector"),
    ("Model Registry", "Versioned models with\nscope: system/project/user"),
    ("Doctor Mode", "Auto-diagnosis with\n8 health checks"),
]

for i, (title, desc) in enumerate(features):
    x = Inches(0.5 + (i % 2) * 6.2)
    y = Inches(1.5 + (i // 2) * 2.5)
    
    box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, Inches(5.5), Inches(2))
    box.fill.solid()
    box.fill.fore_color.rgb = UMRS_NAVY
    box.line.fill.background()
    
    add_textbox(slide, x + Inches(0.2), y + Inches(0.2), Inches(5), Inches(0.6),
                title, font_size=20, font_color=UMRS_TEAL, bold=True)
    
    add_textbox(slide, x + Inches(0.2), y + Inches(0.9), Inches(5), Inches(1),
                desc, font_size=14, font_color=UMRS_WHITE)

add_footer(slide, 7, 10)

# ========== SLIDE 8: Demo Results ==========
print("Slide 8/10: Demo Results")
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(slide, UMRS_WHITE)
add_header_bar(slide, "Demo Results")

# Results table
add_textbox(slide, Inches(0.8), Inches(1.5), Inches(4), Inches(0.5),
            "Feature", font_size=16, font_color=UMRS_NAVY, bold=True)
add_textbox(slide, Inches(5), Inches(1.5), Inches(3), Inches(0.5),
            "Status", font_size=16, font_color=UMRS_NAVY, bold=True)
add_textbox(slide, Inches(8), Inches(1.5), Inches(4.5), Inches(0.5),
            "Details", font_size=16, font_color=UMRS_NAVY, bold=True)

results = [
    ("Server Connectivity", "✓ Online", "data0.ai.r-ccs.riken.jp"),
    ("S3/MinIO Storage", "✓ Operational", "6 buckets configured"),
    ("System Models", "✓ 35 models", "5 embeddings, 3 vision"),
    ("RAG Service", "✓ Available", "PostgreSQL + pgvector"),
    ("Workflow Engine", "✓ Operational", "Multi-step DAG execution"),
    ("Job System", "✓ Responsive", "Slurm integration active"),
    ("Doctor Mode", "✓ Working", "8 health checks pass"),
]

for i, (feature, status, details) in enumerate(results):
    y = Inches(2.2 + i * 0.6)
    
    # Status color
    status_color = UMRS_GREEN if "✓" in status else UMRS_RED
    status_box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                         Inches(5), y, Inches(2.5), Inches(0.4))
    status_box.fill.solid()
    status_box.fill.fore_color.rgb = status_color
    status_box.line.fill.background()
    add_textbox(slide, Inches(5.1), y + Inches(0.02), Inches(2.3), Inches(0.4),
                status, font_size=12, font_color=UMRS_WHITE, alignment=PP_ALIGN.CENTER)
    
    add_textbox(slide, Inches(0.8), y, Inches(4), Inches(0.5),
                feature, font_size=14, font_color=UMRS_GRAY)
    
    add_textbox(slide, Inches(8), y, Inches(4.5), Inches(0.5),
                details, font_size=13, font_color=UMRS_GRAY)

add_footer(slide, 8, 10)

# ========== SLIDE 9: Security & Governance ==========
print("Slide 9/10: Security & Governance")
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(slide, UMRS_WHITE)
add_header_bar(slide, "Security & Governance")

security_items = [
    "JWT-based authentication with role-based access control",
    "S3 presigned URLs — time-limited, unguessable, auditable",
    "Bucket alias system — isolated data per user/project",
    "Model scopes (system/project/user) — granular access control",
    "RAG content scanning — automatic PII/credential detection",
    "EgressGuard — MCP network access with domain allowlisting",
    "Complete audit trail — every action logged & searchable",
    "No model downloads inside inference containers",
    "HTTP-only internal network — no SSL between components"
]
add_bullet_list(slide, Inches(0.8), Inches(1.5), Inches(11), Inches(5.5),
                security_items, font_size=16, font_color=UMRS_NAVY,
                bullet_char="🔒")

add_footer(slide, 9, 10)

# ========== SLIDE 10: Conclusion ==========
print("Slide 10/10: Conclusion & Q&A")
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(slide, UMRS_NAVY)

add_textbox(slide, Inches(2), Inches(2), Inches(9), Inches(1.5),
            "Thank You", font_size=64, font_color=UMRS_WHITE, bold=True,
            alignment=PP_ALIGN.CENTER)

add_textbox(slide, Inches(2), Inches(3.5), Inches(9), Inches(1),
            "Questions & Discussion", font_size=32, font_color=UMRS_TEAL,
            alignment=PP_ALIGN.CENTER)

add_textbox(slide, Inches(2), Inches(5), Inches(9), Inches(1.5),
            "UMRS Platform • Production Ready\n"
            "Unified Multimodal Retrieval System",
            font_size=20, font_color=UMRS_WHITE, alignment=PP_ALIGN.CENTER)

add_footer(slide, 10, 10)

# Save
output_file = OUTPUT_DIR / "demo_day.pptx"
prs.save(output_file)
print("")
print(f"✓ Generated: {output_file}")
print(f"✓ File size: {output_file.stat().st_size / 1024:.0f} KB")
print("")
print("Next steps:")
print("  1. Open demo_day.pptx in PowerPoint")
print("  2. Add actual screenshots from demo runs")
print("  3. Customize colors to match AI4S branding")
print("  4. Add speaker notes to each slide")
