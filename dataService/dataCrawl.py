import json
from bs4 import BeautifulSoup
import re

with open("./crawService/dataCrawl.json", "r", encoding='utf-8') as f:
    crawl_status = json.load(f)

data = crawl_status
print("Data: ", data)

# Hàm làm sạch markdown bằng BeautifulSoup và regex
def clean_markdown(markdown_text):
    # Dùng BeautifulSoup để loại bỏ các thẻ HTML không cần thiết
    soup = BeautifulSoup(markdown_text, "html.parser")

    # Loại bỏ các thẻ a (liên kết)
    for a_tag in soup.find_all('a'):
        a_tag.decompose()

    # Loại bỏ các ký tự markdown như #, *, v.v.
    clean_text = re.sub(r'[##*`]', '', soup.text)

    # Trả về văn bản sạch
    return clean_text.strip()


# Hàm trích xuất và làm sạch dữ liệu
def extract_and_clean(data):
    extracted_data = []

    # Duyệt qua từng mục trong danh sách "data"
    for entry in data["data"]:
        markdown_text = entry.get("markdown", "")
        metadata = entry.get("metadata", {})

        # Làm sạch markdown
        clean_text = clean_markdown(markdown_text)

        # Tạo cấu trúc dữ liệu mới sau khi làm sạch
        cleaned_entry = {
            "title": metadata.get("ogTitle", ""),
            "description": metadata.get("description", ""),
            "keywords": metadata.get("keywords", ""),
            "content": clean_text,
            "sourceURL": metadata.get("sourceURL", "")
        }
        extracted_data.append(cleaned_entry)

    return extracted_data


# Gọi hàm extract_and_clean với dữ liệu của bạn
cleaned_data = extract_and_clean(data)

# Chuyển kết quả sang JSON
cleaned_json = json.dumps(cleaned_data, ensure_ascii=False, indent=2)

# Xuất dữ liệu sạch ra file JSON
with open("./formatData/data.json", "w", encoding='utf-8') as f:
    f.write(cleaned_json)

print("Dữ liệu đã được làm sạch và lưu vào data.json")
