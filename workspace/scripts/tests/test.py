from pathlib import Path
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from docling_core.types.doc import PictureItem

input_pdf = Path("/Users/chrismingard/Kernel/PDF/tests/sample.pdf")
assert input_pdf.exists(), f"File not found: {input_pdf}"

out_dir = Path("extracted_images")
out_dir.mkdir(parents=True, exist_ok=True)

opts = PdfPipelineOptions()
opts.do_ocr = False
opts.generate_page_images = True
opts.generate_picture_images = True
opts.images_scale = 2.0
# You can optionally disable table structure if you don’t need it:
opts.do_table_structure = False

# Use DocumentConverter (or directly StandardPdfPipeline if you prefer)
converter = DocumentConverter(
    format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts, pipeline_cls=StandardPdfPipeline)}
)
result = converter.convert(str(input_pdf))
doc = result.document
stem = input_pdf.stem

for _, page in doc.pages.items():
    page.image.pil_image.save(out_dir / f"{stem}_page_{page.page_no}.png", "PNG")

idx = 0
for element, _ in doc.iterate_items():
    if isinstance(element, PictureItem):
        idx += 1
        img = element.get_image(doc)
        img.save(out_dir / f"{stem}_figure_{idx}.png", "PNG")

print(f"Extracted {idx} cropped figures + full page images → {out_dir}")
