import gradio as gr
from langchain_ollama import OllamaLLM
from langchain.prompts import ChatPromptTemplate

# 1. 初始化模型 (請確保 Ollama 服務已啟動並拉取了 gemma3:latest 模型)
# 運行指令: ollama pull gemma3:latest
# 運行指令: ollama serve
try:
    model = OllamaLLM(model="gemma3:latest")
    print("=== Ollama 模型設定完成 ===")
    print(f"使用模型: gemma3:latest, 類型: {type(model)}")
except Exception as e:
    # 如果 Ollama 服務沒有運行或模型不存在，提供錯誤訊息
    print(f"!!! 警告：初始化 OllamaLLM 失敗。請檢查 Ollama 服務是否運行並拉取了 'gemma3:latest' 模型。錯誤: {e}")
    # 設置一個假的 model 避免程式崩潰，但翻譯功能將會失效
    model = None

# 2. 多變數的複雜模板
COMPLEX_TEMPLATE = """
你是一位專業的{target_language}翻譯家，專精於{domain}領域。
請將以下{source_language}文本翻譯成{target_language}，並確保：
1. 保持原文的語氣和風格
2. 使用專業術語
3. 符合{target_language}的語言習慣

{source_language}文本：{text}
{target_language}翻譯：
"""
chat_prompt_template = ChatPromptTemplate.from_template(COMPLEX_TEMPLATE)

# 3. 定義 Gradio 呼叫函式
def professional_translation(
    source_language: str, 
    target_language: str, 
    domain: str, 
    text: str
) -> str:
    """
    接收使用者輸入的參數，格式化 Prompt，並呼叫 Ollama 模型進行翻譯。
    """
    if model is None:
        return "錯誤：LLM 模型初始化失敗。請檢查你的 Ollama 服務！"
    
    try:
        # 格式化 Prompt
        formatted_prompt = chat_prompt_template.format(
            source_language=source_language,
            target_language=target_language, 
            domain=domain,
            text=text
        )
        
        # 呼叫模型
        response = model.invoke(formatted_prompt)
        return response
    
    except Exception as e:
        return f"翻譯過程中發生錯誤: {e}"


# 4. 建立 Gradio 介面
# 設定輸入元件
input_components = [
    gr.Dropdown(
        label="原始語言 (Source Language)", 
        choices=["英文", "繁體中文", "簡體中文", "日文"], 
        value="英文", 
        interactive=True
    ),
    gr.Dropdown(
        label="目標語言 (Target Language)", 
        choices=["繁體中文", "英文", "簡體中文", "日文"], 
        value="繁體中文", 
        interactive=True
    ),
    gr.Textbox(
        label="專業領域 (Domain)", 
        placeholder="例如：商業, 法律, 科技, 文學...", 
        value="商業", 
        interactive=True
    ),
    gr.Textbox(
        label="待翻譯文本 (Text to Translate)", 
        placeholder="請輸入您想要翻譯的文本...", 
        lines=5,
        value="Harrods says customers' data stolen in IT breach.",
        interactive=True
    )
]

# 建立 Interface
iface = gr.Interface(
    fn=professional_translation,
    inputs=input_components,
    outputs=gr.Textbox(label="翻譯結果 (Translation Result)", lines=8),
    title="🤖 Ollama + Gradio 專業領域翻譯機",
    description="利用 LangChain Prompt Template 和 Ollama (Gemma 3) 進行多變數控制的專業翻譯。您可以設定原始語言、目標語言和專業領域，來獲取更精準的結果。",
    allow_flagging="never", # 關閉 Gradio 的資料蒐集按鈕
    theme=gr.themes.Soft(), # 使用一個美觀的主題
)

# 啟動 Gradio 介面
iface.launch()