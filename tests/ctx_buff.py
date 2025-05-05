import lib.redis.context_buffer as ctx_buff

print(f"buffer: {ctx_buff.buffer}")
for i in ctx_buff.buffer:
    print(f"ready: {ctx_buff.get_ready_items()}")
    for j in ctx_buff.get_ready_items():
        ctx_buff.mark_processed(j)

    print(f"processed: {ctx_buff.processed}")
