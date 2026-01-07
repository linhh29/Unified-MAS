"""
测试 OpenAI API key 并返回可用的模型列表
"""
from typing import List, Dict, Optional
from openai import OpenAI
from openai import AuthenticationError, APIError


def get_available_models(api_key: str, api_base: Optional[str] = None) -> List[Dict[str, str]]:
    """
    给定一个 OpenAI API key，返回可用的模型列表
    
    Args:
        api_key: OpenAI API key
        api_base: 可选的 API base URL（用于兼容其他 OpenAI 兼容的 API）
    
    Returns:
        包含模型信息的字典列表，每个字典包含 'id' 和 'object' 等字段
    
    Raises:
        AuthenticationError: API key 无效
        APIError: API 调用失败
    """
    try:
        # 创建 OpenAI 客户端
        client_kwargs = {"api_key": api_key}
        if api_base:
            client_kwargs["base_url"] = api_base
        
        client = OpenAI(**client_kwargs)
        
        # 获取模型列表
        models = client.models.list()
        
        # 提取模型信息
        model_list = []
        for model in models.data:
            model_info = {
                "id": model.id,
                "object": model.object,
                "created": model.created,
                "owned_by": model.owned_by,
            }
            model_list.append(model_info)
        
        return model_list
    
    except AuthenticationError as e:
        raise AuthenticationError(f"API key 无效或认证失败: {str(e)}")
    except APIError as e:
        raise APIError(f"API 调用失败: {str(e)}")
    except Exception as e:
        raise Exception(f"获取模型列表时发生错误: {str(e)}")


def get_available_model_names(api_key: str, api_base: Optional[str] = None) -> List[str]:
    """
    给定一个 OpenAI API key，返回可用的模型名称列表（简化版本）
    
    Args:
        api_key: OpenAI API key
        api_base: 可选的 API base URL
    
    Returns:
        模型 ID 的字符串列表
    """
    models = get_available_models(api_key, api_base)
    return [model["id"] for model in models]


if __name__ == "__main__":
    import sys
    
    # 从命令行参数或环境变量获取 API key
    if len(sys.argv) > 1:
        api_key = sys.argv[1]
    else:
        import os
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("错误: 请提供 OpenAI API key")
            print("用法: python test_openai.py <api_key>")
            print("或者设置环境变量 OPENAI_API_KEY")
            sys.exit(1)
    
    # 可选：从命令行获取 API base URL
    api_base = None
    if len(sys.argv) > 2:
        api_base = sys.argv[2]
    
    try:
        print(f"正在获取可用模型列表...")
        print(f"使用 API key: {api_key[:10]}...{api_key[-4:]}")
        if api_base:
            print(f"使用 API base: {api_base}")
        print("-" * 60)
        
        # 获取模型列表
        models = get_available_models(api_key, api_base)
        
        print(f"\n找到 {len(models)} 个可用模型:\n")
        
        # # 打印模型信息
        # for i, model in enumerate(models, 1):
        #     if 'gpt-5' in model['id']:
        #         print(f"{i}. {model['id']}")
        #         print(f"   对象类型: {model['object']}")
        #         print(f"   创建时间: {model['created']}")
        #         print(f"   所有者: {model['owned_by']}")
        #         print()
            # else:
            #     continue
            # print(f"{i}. {model['id']}")
            # print(f"   对象类型: {model['object']}")
            # print(f"   创建时间: {model['created']}")
            # print(f"   所有者: {model['owned_by']}")
            # print()
        
        # 也打印简化的模型名称列表
        print("-" * 60)
        print("\n模型名称列表:")
        model_names = get_available_model_names(api_key, api_base)
        for name in model_names:
            if 'gpt-5' in name:
                print(f"  - {name}")
        
    except AuthenticationError as e:
        print(f"认证错误: {e}")
        sys.exit(1)
    except APIError as e:
        print(f"API 错误: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)

    client = OpenAI(
        api_key=api_key
    )
    response = client.chat.completions.create(
        model='gpt-5-mini',
        messages=[{'role': 'user', 'content': 'Tell me a story about a cat.'}],
        max_completion_tokens=32768,
        reasoning_effort='medium',
    )
    print(response.choices[0].message.content)