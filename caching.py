from typing import Any, Tuple


def invalidate_cache_entry(function_handle, function_arguments: Tuple[Any], invalidate=True):
    if not invalidate:
        return
    is_cache_hit = function_handle.check_call_in_cache(*function_arguments)
    if not is_cache_hit:
        print("Invalidate cache entry: failed, missing")
        return
    args_id = function_handle._get_args_id(*function_arguments)
    call_id = [function_handle.func_id, args_id]
    function_handle.store_backend.clear_item(call_id)
    print("Invalidate cache entry: successful")
