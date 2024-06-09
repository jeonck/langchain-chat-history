# 패키지 설치
# pip install --upgrade --quiet langchain langchain-openai langchain-community
# https://python.langchain.com/v0.2/docs/how_to/chatbots_memory/

# 환경 변수 설정
import os
import dotenv
dotenv.load_dotenv()

api_key = os.environ["OPENAI_API_KEY"]

# 필수 모듈 임포트
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnablePassthrough

# 모델 초기화
chat = ChatOpenAI(model="gpt-4o", temperature=0.2)

# 프롬프트 템플릿 설정
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Answer all questions to the best of your ability."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)

# 체인 설정
chain = prompt | chat

# 메모리 히스토리 클래스 초기화
demo_ephemeral_chat_history = ChatMessageHistory()

# 메시지 트리밍 함수 정의
def trim_messages(chain_input):
    stored_messages = demo_ephemeral_chat_history.messages
    if len(stored_messages) <= 2:
        return False

    demo_ephemeral_chat_history.clear()

    for message in stored_messages[-2:]:
        demo_ephemeral_chat_history.add_message(message)

    return True

# 요약 생성 함수 정의
def summarize_messages(chain_input):
    stored_messages = demo_ephemeral_chat_history.messages
    if len(stored_messages) == 0:
        return False
    summarization_prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "Distill the above chat messages into a single summary message. Include as many specific details as you can."),
        ]
    )
    summarization_chain = summarization_prompt | chat

    summary_message = summarization_chain.invoke({"chat_history": stored_messages})

    demo_ephemeral_chat_history.clear()
    demo_ephemeral_chat_history.add_message(summary_message)

    return True

# 자동 이력 관리 체인 설정
chain_with_message_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: demo_ephemeral_chat_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

# 트리밍과 요약을 포함한 체인 설정
chain_with_trimming_and_summarization = (
    RunnablePassthrough.assign(messages_trimmed=trim_messages)
    | RunnablePassthrough.assign(messages_summarized=summarize_messages)
    | chain_with_message_history
)

# 체인 호출 예제
response = chain_with_trimming_and_summarization.invoke(
    {"input": "Translate this sentence from English to French: I love programming."},
    {"configurable": {"session_id": "demo"}}
)
print(response.content)

response = chain_with_trimming_and_summarization.invoke(
    {"input": "What did I just ask you?"},
    {"configurable": {"session_id": "demo"}}
)
print(response.content)

# 현재 히스토리 출력
print(demo_ephemeral_chat_history.messages)
