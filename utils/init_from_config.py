import inspect

import inspect


def auto_init(cls, munch_obj):
    """匹配类中初始化方法和config中储存的对应值，构建字典储存匹配的config键值对"""
    # 获取类及其所有父类的 __init__ 方法的参数
    all_params = set()
    for c in cls.__mro__:
        if '__init__' in c.__dict__:
            init_signature = inspect.signature(c.__init__)
            all_params.update(init_signature.parameters.keys())

    # 移除 'self' 参数
    all_params.discard('self')

    # 过滤 munch_obj 中与所有参数匹配的键值对
    filtered_params = {
        key: value for key, value in munch_obj.items()
        if key in all_params
    }

    return filtered_params



