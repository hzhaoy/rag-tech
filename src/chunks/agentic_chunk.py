import os
import re

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.workflow import Workflow, StartEvent, StopEvent, step
from llama_index.llms.openai_like import OpenAILike
from llama_index.core.base.llms.types import ChatMessage, MessageRole, TextBlock

from src.config import get_logger, PROJECT_ROOT
from src.models.chat_models import create_chat_model

from .types import DocumentEvent, SentenceEvent, LLMDecisionEvent


logger = get_logger(__name__)


class DocumentChunkingWorkflow(Workflow):
    def __init__(self, llm):
        super().__init__()
        self.llm: OpenAILike = llm
        self.sentence_parser = SentenceSplitter(chunk_size=100, chunk_overlap=0)

    @step
    async def input_node(self, ev: StartEvent) -> DocumentEvent:
        """输入节点：接收原始文本并创建Document对象"""
        return DocumentEvent(document=Document(text=ev.input_text))

    @step
    async def sentence_splitting_node(self, ev: DocumentEvent) -> SentenceEvent:
        """句子分割节点：将文档分割成单个句子"""
        # sentence_nodes = self.sentence_parser.get_nodes_from_documents([ev.document])
        # sentences = [node.text for node in sentence_nodes]
        sentences = re.split(r"(?<=[.。?!])\s+", ev.document.text)
        sentences = [x.strip() for i, x in enumerate(sentences) if x]
        logger.info(f"sentence_splitting_node len(sentences): {len(sentences)}")
        return SentenceEvent(sentences=sentences)

    @step
    async def llm_decision_node(self, ev: SentenceEvent) -> LLMDecisionEvent:
        """LLM决策节点：决定如何将句子分组"""
        sentences = ev.sentences
        if not sentences:
            return LLMDecisionEvent(sentence_groups=[])

        grouped_sentences = []
        current_group = [sentences[0]]

        for next_sentence in sentences[1:]:
            # 清理句子文本
            current_chunk = " ".join(current_group)
            current_chunk = re.sub(r"\s+", " ", current_chunk).strip()
            next_sentence_clean = re.sub(r"\s+", " ", next_sentence).strip()

            # 使用LLM做出决策
            prompt = f"""
            Consider the following text chunk and next sentence:
            Current chunk: '{current_chunk}'
            Next sentence: '{next_sentence_clean}'

            Should the next sentence belong to the current chunk based on topic continuity and semantic coherence?
            Answer with 'yes' or 'no' only.
            """

            response = self.llm.chat(
                messages=[
                    ChatMessage(role=MessageRole.USER, blocks=[TextBlock(text=prompt)]),
                ],
            )
            decision = response.message.content

            if (
                decision is not None
                and decision.startswith("yes")
                and sum(len(s) for s in current_group) + len(next_sentence_clean) <= 500
            ):
                logger.info(
                    f"Appending to current group: {next_sentence_clean[:30]}..."
                )
                current_group.append(next_sentence_clean)
            else:
                logger.info(f"Starting new group with: {next_sentence_clean[:30]}...")
                grouped_sentences.append(current_group)
                current_group = [next_sentence_clean]

        if current_group:
            grouped_sentences.append(current_group)

        return LLMDecisionEvent(sentence_groups=grouped_sentences)

    @step
    async def chunking_node(self, ev: LLMDecisionEvent) -> StopEvent:
        """分块节点：组织最终的分块结果"""
        return StopEvent(
            result=[
                {
                    "chunk": " ".join(sentences),
                    "sentences": sentences,
                    "chunk_id": f"chunk_{i + 1}",
                }
                for i, sentences in enumerate(ev.sentence_groups)
            ]
        )


async def main():
    # 初始化LLM
    llm = create_chat_model(
        model=os.getenv("MODEL", "kimi-k2-250905"),
        api_key=os.getenv("OPENAI_API_KEY"),
        api_base=os.getenv("OPENAI_BASE_URL"),
    )

    # 示例文档文本
    with open(os.path.join(PROJECT_ROOT, "data", "chunk_demo.txt"), "r") as f:
        document_text = f.read()

    # 创建并运行工作流
    w = DocumentChunkingWorkflow(llm)
    result = await w.run(input_text=document_text)

    # 打印结果
    for chunk in result:
        print(f"Chunk ID: {chunk['chunk_id']}")
        print(f"Content: {chunk['chunk']}")
        print(f"Number of sentences: {len(chunk['sentences'])}")
        print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
