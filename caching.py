from run import run


def invalidate_run_cache_entry(*function_arguments):
    is_cache_hit = run.check_call_in_cache(*function_arguments)
    if not is_cache_hit:
        return
    args_id = run._get_args_id(*function_arguments)
    call_id = [run.func_id, args_id]
    run.store_backend.clear_item(call_id)
