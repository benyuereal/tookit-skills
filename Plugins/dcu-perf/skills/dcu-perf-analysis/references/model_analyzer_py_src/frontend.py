

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import subprocess
import sys

from .model_factory import ModelFactory, ModelType
from .parallel_config import ParallelConfig


class LLMPerfAnalyzer:
    """LLM性能分析器前端类"""
    
    def __init__(self, model_type: str, config_path: Optional[str] = None):
        """
        初始化分析器
        
        Args:
            model_type: 模型类型，如 'deepseek_v3', 'llama2'
            config_path: 配置文件路径（可选）
        """
        self.model_type = ModelType(model_type.lower())
        self.config_path = config_path
        
        self.config = ModelFactory.get_model_config(self.model_type, config_path)
        
        self.parallel_config = ParallelConfig()
        
        self._check_backend()
    
    def _check_backend(self):
        """检查C++后端是否存在"""
        backend_path = Path(__file__).parent.parent.parent / "build" / "model_analyzer"
        if not backend_path.exists():
            print(f"警告: C++后端未找到，尝试编译...")
            self._compile_backend()
    
    def _compile_backend(self):
        """编译C++后端"""
        project_root = Path(__file__).parent.parent.parent
        build_script = project_root / "scripts" / "build.sh"
        
        if build_script.exists():
            try:
                subprocess.run([str(build_script)], check=True, cwd=project_root)
                print("C++后端编译成功")
            except subprocess.CalledProcessError as e:
                print(f"编译失败: {e}")
                raise RuntimeError("无法编译C++后端")
        else:
            raise RuntimeError("编译脚本未找到")
    
    def set_parallel_config(self, tp_size: int = 1, pipeline_size: int = 1):
        """设置并行配置"""
        self.parallel_config.tp_size = tp_size
        self.parallel_config.pipeline_size = pipeline_size
    
    def analyze(self, batch_size: int = 1, seq_len: int = 1024, 
                decode_steps: int = 100) -> Dict[str, Any]:
        """
        分析模型性能
        
        Args:
            batch_size: 批大小
            seq_len: 序列长度
            decode_steps: 解码步数
            
        Returns:
            性能分析结果
        """
        print(f"开始分析 {self.model_type.value} 模型性能...")
        print(f"批大小: {batch_size}, 序列长度: {seq_len}")
        print(f"并行配置: {self.parallel_config}")
        
        backend_path = Path(__file__).parent.parent.parent / "build" / "model_analyzer"
        
        cmd = [
            str(backend_path),
            self.config_path if self.config_path else str(self.model_type.value),
            str(batch_size),
            str(seq_len),
            str(self.parallel_config.tp_size),
            str(decode_steps)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(result.stdout)
            
            if result.stderr:
                print(f"警告: {result.stderr}")
            
            return self._parse_results(result.stdout)
            
        except subprocess.CalledProcessError as e:
            print(f"分析失败: {e}")
            print(f"标准错误: {e.stderr}")
            raise
    
    def _parse_results(self, output: str) -> Dict[str, Any]:
        """解析C++后端的输出结果"""
        lines = output.strip().split('\n')
        results = {
            "prefill": {},
            "decode": {},
            "throughput": {},
            "efficiency": {}
        }
        
        current_section = None
        for line in lines:
            line = line.strip()
            
            if "Prefill阶段" in line:
                current_section = "prefill"
            elif "Decoder阶段" in line:
                current_section = "decode"
            elif "吞吐量分析" in line:
                current_section = "throughput"
            elif "计算效率" in line:
                current_section = "efficiency"
            elif current_section and ":" in line:
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip()
                
                if current_section == "throughput":
                    if "tokens/s" in value:
                        num = value.split()[0]
                        results[current_section][key] = float(num)
                elif current_section == "efficiency":
                    if "TFLOPs" in value:
                        num = value.split()[0]
                        results[current_section][key] = float(num)
                else:
                    try:
                        if "ms" in value:
                            num = value.split()[0]
                            results[current_section][key] = float(num)
                        elif "TFLOPs" in value:
                            num = value.split()[0]
                            results[current_section][key] = float(num)
                        else:
                            results[current_section][key] = value
                    except:
                        results[current_section][key] = value
        
        return results
    
    def analyze_sensitivity(self, batch_sizes: list, seq_lens: list) -> Dict[str, Any]:
        """
        分析性能对批大小和序列长度的敏感性
        
        Args:
            batch_sizes: 批大小列表
            seq_lens: 序列长度列表
            
        Returns:
            敏感性分析结果
        """
        results = {
            "prefill": {},
            "decode": {}
        }
        
        for bs in batch_sizes:
            for sl in seq_lens:
                print(f"\n分析 batch_size={bs}, seq_len={sl}...")
                try:
                    result = self.analyze(bs, sl, decode_steps=1)
                    
                    key = f"bs{bs}_sl{sl}"
                    results["prefill"][key] = result["prefill"].get("Total Latency", 0)
                    results["decode"][key] = result["decode"].get("Total Latency", 0)
                    
                except Exception as e:
                    print(f"分析失败: {e}")
                    results["prefill"][key] = None
                    results["decode"][key] = None
        
        return results
    
    def get_model_info(self) -> str:
        """获取模型信息"""
        return ModelFactory.get_model_info(self.config)
    
    def save_config(self, filepath: str):
        """保存配置到文件"""
        with open(filepath, 'w') as f:
            json.dump(self.config, f, indent=2)
        print(f"配置已保存到: {filepath}")


def main():
    """命令行入口点"""
    import argparse
    
    parser = argparse.ArgumentParser(description="LLM性能分析工具")
    parser.add_argument("--model", type=str, default="deepseek_v3",
                       help="模型类型 (deepseek_v3, llama2, llama3, mistral, qwen)")
    parser.add_argument("--config", type=str, help="配置文件路径")
    parser.add_argument("--batch-size", "-b", type=int, default=1, help="批大小")
    parser.add_argument("--seq-len", "-s", type=int, default=1024, help="序列长度")
    parser.add_argument("--tp-size", "-t", type=int, default=1, help="Tensor并行大小")
    parser.add_argument("--decode-steps", "-d", type=int, default=100, help="解码步数")
    parser.add_argument("--output", "-o", type=str, help="输出文件路径")
    
    args = parser.parse_args()
    
    analyzer = LLMPerfAnalyzer(args.model, args.config)
    analyzer.set_parallel_config(tp_size=args.tp_size)
    
    print("\n" + analyzer.get_model_info())
    
    results = analyzer.analyze(args.batch_size, args.seq_len, args.decode_steps)
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n结果已保存到: {args.output}")


if __name__ == "__main__":
    main()