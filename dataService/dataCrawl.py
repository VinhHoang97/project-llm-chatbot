import json
from bs4 import BeautifulSoup
import re

with open("./crawService/dataCrawl.json", "r", encoding='utf-8') as f:
    crawl_status = json.load(f)

data = crawl_status
# print("Data: ", data)

# Hàm làm sạch markdown bằng BeautifulSoup và regex
def clean_markdown(markdown_text):
    # Dùng BeautifulSoup để loại bỏ các thẻ HTML không cần thiết
    soup = BeautifulSoup(markdown_text, "html.parser")

    # Loại bỏ các thẻ a (liên kết)
    for a_tag in soup.find_all('a'):
        a_tag.decompose()

    # Loại bỏ các ký tự markdown như #, *, v.v.
    clean_text = ""

    # Tìm tất cả các phần tử h2 theo đường dẫn CSS
    elements_h2 = soup.select("#main-detail > h2")
    elements_h2_video = soup.select("#main-detail > div.media__detail-top > div > div > h2")
    # Tìm tất cả các phần tử p theo đường dẫn CSS
    elements_p = soup.select("#main-detail > div.detail-cmain > div.detail-content > p")
    
    for idx, element in enumerate(elements_h2):
        # print(f"Dữ liệu từ phần tử h2 [{idx}]:", element.get_text())
        clean_text += element.get_text()
    
    for idx, element in enumerate(elements_h2_video):
        # print(f"Dữ liệu từ phần tử h2 [{idx}]:", element.get_text())
        clean_text += element.get_text()
        
    # Lấy dữ liệu từ các phần tử p nếu tìm thấy
    for idx, element in enumerate(elements_p):
        
        text = ""
        for content in element.contents:
            if content.name is not None:  # Nếu content là một thẻ (ví dụ <i>, <b>)
                text += " "
            else:
                text += content
                
        text = ' '.join(text.split())
        # print(f"Dữ liệu từ phần tử p [{idx}]:", text)
        clean_text = clean_text + " " + text

    # Trả về văn bản sạch
    return clean_text.strip()

def clean_text(text):
    # Loại bỏ các đường link
    text = re.sub(r'http\S+', '', text)
    # Loại bỏ ký tự xuống dòng
    text = text.replace('\n\n', '')
    return text.strip()


# Hàm trích xuất và làm sạch dữ liệu
def extract_and_clean(data):
    extracted_data = []

    # Duyệt qua từng mục trong danh sách "data"
    for entry in data["data"]:
        if (entry is None):
            continue
        markdown_text = entry.get("html", "")
        metadata = entry.get("metadata", {})

        # Làm sạch markdown
        clean_text_str = clean_markdown(markdown_text)
        if (len(clean_text_str) == 0):
            continue
        # Tạo cấu trúc dữ liệu mới sau khi làm sạch
        cleaned_entry = {
            "title": metadata.get("ogTitle", ""),
            "description": metadata.get("description", ""),
            "keywords": metadata.get("keywords", ""),
            "content": clean_text(clean_text_str),
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
