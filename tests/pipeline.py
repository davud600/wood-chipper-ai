import threading
import time
import fitz

# from config import pages_to_append
# from lib.doc_tools.documents import convert_pdf_page_to_image, get_image_contents

# from utils import split_arr
from lib.redis import redis
from core.pipeline import process_pages_pipeline


if __name__ == "__main__":
    # temp: del redis stuff.
    redis.flushall()

    # t0 = time.time()
    #
    # print("loading tokenizer...")
    # tokenizer = AutoTokenizer.from_pretrained(
    #     "allenai/longformer-base-4096", device="cuda"
    # )
    #
    # t1 = time.time()
    # print(f"load tokenizer time: {t1-t0}")
    #
    # print("loading model...")
    # model = SplitterModel().to("cuda")
    # model.eval()
    # model.load_state_dict(
    #     torch.load(SPLITTER_MODEL_PATH, weights_only=False, map_location="cuda")
    # )
    #
    # t2 = time.time()
    # print(f"load model time: {t2-t1}")

    document_context = {
        "file_path": "/home/davud/wood-chipper-ai/test.pdf",
        "file_name": "test.pdf",
        "document_id": 1,
        "transaction_id": 1,
        "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOjEsImlzQWRtaW4iOnRydWUsIm5hbWUiOiJBZG1pbiIsImVtYWlsIjoiYWRtaW5AZ21haWwuY29tIiwiaWF0IjoxNzQzOTQyNDg1LCJleHAiOjE3NDM5NzEyODV9.oXn4HzMeAtIZZu3POZI1_w0BGY1ZZ5HRUpofB2GCERA",
    }
    doc = fitz.open(document_context["file_path"])
    document_pages = len(doc)

    real_contents = []
    # for page in range(document_pages):
    #     real_contents += [
    #         get_image_contents(
    #             convert_pdf_page_to_image(document_context["file_name"], page, doc)  # type: ignore
    #         )
    #     ]

    print("\nstarting pipeline...")
    t3 = time.time()

    try:

        def thread_target():
            process_pages_pipeline(
                pages=document_pages,
                document_context=document_context,
            )

        # def thread_target1():
        #     process_pages_pipeline(
        #         pages=document_pages,
        #         document_context=document_context1,
        #     )

        thread = threading.Thread(
            target=thread_target,
        )
        thread.start()
        # thread1 = threading.Thread(
        #     target=thread_target1,
        # )
        # thread1.start()
        thread.join()
        # thread1.join()
    except Exception as e:
        print(e)

    t4 = time.time()
    print(f"\npipeline time: {t4-t3}")

    for page in range(document_pages):
        raw = redis.get(f"page_content:{document_context['document_id']}:{page}")
        redis_content = raw.decode("utf-8") if raw else ""  # type: ignore
        print(f"page {page} redis content: {redis_content[:100]}")
        # print(
        #     f"page {page} redis content: {redis_content[:25]} | page {page} real content: {real_contents[page][:25]}"
        # )

    # for page in range(document_pages):
    #     raw = redis.get(f"page_content:{document_context1['document_id']}:{page}")
    #     redis_content = raw.decode("utf-8") if raw else ""  # type: ignore
    #     print(f"page {page} redis content: {redis_content[:25]}")
    #     # print(
    #     #     f"page {page} redis content: {redis_content[:25]} | page {page} real content: {real_contents[page][:25]}"
    #     # )
