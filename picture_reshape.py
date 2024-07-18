from docx import Document
def resize_images(docx_file, new_width, new_height):
    doc = Document(docx_file)

    for rel in doc.part.rels.values():
        if "image" in rel.reltype:
            image_part = rel.target_part
            image = image_part.blob

            # 修改图片尺寸
            image_part.width = new_width
            image_part.height = new_height

    doc.save('resized_images.docx')
    print("图片已调整大小并保存为 resized_images.docx")

# 调整后的图片尺寸
new_width = 300
new_height = 300

# 输入你的 Word 文档文件名
word_doc = 'your_document.docx'

resize_images(word_doc, new_width, new_height)
