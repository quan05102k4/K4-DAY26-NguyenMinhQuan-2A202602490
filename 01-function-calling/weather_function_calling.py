"""Minh hoạ FUNCTION CALLING thuần với OpenAI SDK.

Tool `get_weather` được định nghĩa schema thủ công VÀ thực thi ngay trong
chính file app này. Model chỉ QUYẾT ĐỊNH gọi tool nào; app mới là nơi chạy.

Cách chạy:
    pip install -r ../requirements.txt
    # Đặt OPENAI_API_KEY trong file .env ở thư mục gốc (tự động được nạp)
    # hoặc trong PowerShell: $env:OPENAI_API_KEY="..."
    python weather_function_calling.py
"""

import json

from dotenv import load_dotenv
from openai import OpenAI

# Nạp biến từ .env (tìm ngược lên các thư mục cha) trước khi tạo client
load_dotenv()

client = OpenAI()

MODEL = "gpt-4o-mini"

SYSTEM_INSTRUCTION = (
    "Bạn là trợ lý thời tiết thân thiện, trả lời bằng tiếng Việt tự nhiên. "
    "Dùng emoji phù hợp (🌧️ 🌤️ 💨 💧). "
    "Tóm tắt ngắn gọn, dễ hiểu, và đưa ra lời khuyên thực tế "
    "(ví dụ: mang ô, mặc áo mỏng, ...)."
)

# 1. App tự định nghĩa schema của tool (định dạng tool của OpenAI)
get_weather_declaration = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Lấy thời tiết hiện tại của một thành phố",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "Tên thành phố"}
            },
            "required": ["city"],
        },
    },
}

TOOLS = [get_weather_declaration]


# 2. App tự thực thi tool (trong thực tế sẽ gọi API thời tiết thật)
def get_weather(city: str) -> str:
    """Trả về thời tiết (mock) của *city*. Dùng làm tool cho model."""
    mock_data = {
        "Hà Nội": {
            "nhiệt_độ": "29°C",
            "thời_tiết": "trời mưa nhẹ",
            "độ_ẩm": "82%",
            "gió": {"hướng": "Đông Nam", "tốc_độ": "12 km/h"},
        },
        "Hồ Chí Minh": {
            "nhiệt_độ": "33°C",
            "thời_tiết": "mưa rào",
            "độ_ẩm": "75%",
            "gió": {"hướng": "Tây Nam", "tốc_độ": "15 km/h"},
        },
        "Đà Nẵng": {
            "nhiệt_độ": "30°C",
            "thời_tiết": "nhiều mây",
            "độ_ẩm": "78%",
            "gió": {"hướng": "Đông", "tốc_độ": "10 km/h"},
        },
    }

    default = {"nhiệt_độ": "28°C", "thời_tiết": "không có dữ liệu chi tiết"}
    return json.dumps({"city": city, **mock_data.get(city, default)}, ensure_ascii=False)


def run(prompt: str) -> str:
    """Gửi *prompt* tới OpenAI, tự động xử lý function calling và trả về câu trả lời cuối."""
    messages: list = [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": prompt},
    ]

    # 3. Gọi model — model quyết định có gọi tool hay không
    resp = client.chat.completions.create(model=MODEL, messages=messages, tools=TOOLS)
    message = resp.choices[0].message

    # 4. Vòng lặp: nếu model yêu cầu tool, app TỰ THỰC THI rồi đưa kết quả trả lại
    while message.tool_calls:
        # Thêm phản hồi của model (chứa các yêu cầu gọi tool) vào lịch sử hội thoại
        messages.append(message)

        for fc in message.tool_calls:
            args = json.loads(fc.function.arguments)
            print(f"  [model yêu cầu] {fc.function.name}({args})")
            result = get_weather(**args)  # <-- app chạy, không phải model
            print(f"  [app thực thi]  -> {result}")
            # Gửi kết quả tool trả về cho model
            messages.append({"role": "tool", "tool_call_id": fc.id, "content": result})

        resp = client.chat.completions.create(model=MODEL, messages=messages, tools=TOOLS)
        message = resp.choices[0].message

    # 5. Model tổng hợp câu trả lời cuối
    return message.content


if __name__ == "__main__":
    question = "Thời tiết Hà Nội và Đà Nẵng hôm nay thế nào?"
    print(f"User: {question}\n")
    print("Trả lời:", run(question))
