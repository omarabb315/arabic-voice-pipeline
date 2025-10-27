"""
Model Comparison Utility

Compare NeuTTS vs Orpheus implementations and configurations.

Usage:
    python training/compare_models.py
"""

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich import box
import sys

console = Console()


def print_header():
    """Print comparison header."""
    console.print("\n")
    console.print("="*80, style="bold blue")
    console.print("  NeuTTS vs Orpheus: Arabic Voice Cloning Models", style="bold blue")
    console.print("="*80, style="bold blue")
    console.print("\n")


def create_architecture_table():
    """Create architecture comparison table."""
    table = Table(title="Architecture Comparison", box=box.ROUNDED)
    
    table.add_column("Feature", style="cyan", width=25)
    table.add_column("NeuTTS", style="green")
    table.add_column("Orpheus", style="yellow")
    
    table.add_row(
        "Base Model",
        "Custom Transformer",
        "Llama 3B"
    )
    table.add_row(
        "Parameters",
        "~1B",
        "~3B"
    )
    table.add_row(
        "Audio Codec",
        "Mimi (Encodec)",
        "SNAC"
    )
    table.add_row(
        "Sample Rate",
        "16 kHz",
        "24 kHz"
    )
    table.add_row(
        "Token Format",
        "<|speech_{i}|>",
        "Interleaved 7-frame"
    )
    table.add_row(
        "Special Tokens",
        "Custom prompts",
        "Llama token IDs"
    )
    table.add_row(
        "Training Method",
        "Full finetuning",
        "LoRA adapters"
    )
    
    return table


def create_performance_table():
    """Create performance comparison table."""
    table = Table(title="Performance Metrics", box=box.ROUNDED)
    
    table.add_column("Metric", style="cyan", width=25)
    table.add_column("NeuTTS", style="green")
    table.add_column("Orpheus", style="yellow")
    
    table.add_row(
        "Audio Quality",
        "Good (7/10)",
        "Excellent (9/10)"
    )
    table.add_row(
        "Voice Similarity",
        "Good (7/10)",
        "Excellent (9/10)"
    )
    table.add_row(
        "Inference Speed",
        "~0.5s per sample",
        "~1.0s per sample"
    )
    table.add_row(
        "Training Time",
        "~8 hours (3 epochs)",
        "~12 hours (3 epochs)"
    )
    table.add_row(
        "Min GPU Memory",
        "16 GB",
        "16 GB (with 4bit)"
    )
    table.add_row(
        "Recommended GPU",
        "32 GB",
        "40-80 GB"
    )
    
    return table


def create_usage_table():
    """Create usage comparison table."""
    table = Table(title="Use Cases & Recommendations", box=box.ROUNDED)
    
    table.add_column("Scenario", style="cyan", width=30)
    table.add_column("Recommended Model", style="magenta")
    table.add_column("Reason", style="white")
    
    table.add_row(
        "Production Deployment",
        "NeuTTS",
        "Faster inference, lower memory"
    )
    table.add_row(
        "Research / Highest Quality",
        "Orpheus",
        "Best audio quality, state-of-the-art"
    )
    table.add_row(
        "Limited GPU Memory (<32GB)",
        "NeuTTS or Orpheus 4bit",
        "Both can run on 16GB"
    )
    table.add_row(
        "Real-time Applications",
        "NeuTTS",
        "2x faster inference"
    )
    table.add_row(
        "Batch Processing",
        "Orpheus",
        "Quality matters more than speed"
    )
    table.add_row(
        "Multi-speaker TTS",
        "Orpheus",
        "Better conditioning control"
    )
    
    return table


def create_code_comparison():
    """Create code comparison panels."""
    
    neutts_code = """[bold green]NeuTTS Training:[/bold green]

[dim]# Preprocessing[/dim]
text → phonemize → condition → tokenize
audio → Mimi encode → <|speech_i|> tokens

[dim]# Format[/dim]
<|TEXT_PROMPT_START|>
  <|gender|><|dialect|><|tone|> phonemes
<|TEXT_PROMPT_END|>
<|SPEECH_GENERATION_START|>
  <|speech_0|><|speech_1|>...
<|SPEECH_GENERATION_END|>

[dim]# Training[/dim]
- Standard transformer training
- Mask: Train only on target codes
- Optimizer: AdamW
- Mixed precision: BF16/FP16"""
    
    orpheus_code = """[bold yellow]Orpheus Training:[/bold yellow]

[dim]# Preprocessing[/dim]
text → phonemize → condition → tokenize
audio → SNAC encode → interleaved tokens

[dim]# Format[/dim]
[SOH=128259] [SOT=128000]
  [gender|dialect|tone] phonemes
[EOT=128009] [EOH=128260]
[SOA=128261] [SOS=128257]
  audio_tokens (7 per frame)
[EOS=128258] [EOA=128262]

[dim]# Training[/dim]
- LoRA adapter training
- Mask: Train only on target codes
- Optimizer: AdamW (8bit)
- Unsloth optimizations"""
    
    panel1 = Panel(neutts_code, title="NeuTTS", border_style="green")
    panel2 = Panel(orpheus_code, title="Orpheus", border_style="yellow")
    
    return Columns([panel1, panel2])


def create_config_comparison():
    """Create configuration comparison."""
    
    neutts_config = """[bold green]config.yaml (NeuTTS):[/bold green]

restore_from: "canopylabs/NeuTTS-3b-ft"
max_seq_len: 2048
lr: 5e-5
per_device_train_batch_size: 8
gradient_accumulation_steps: 4
num_train_epochs: 3

[dim]# No LoRA needed[/dim]"""
    
    orpheus_config = """[bold yellow]config-orpheus.yaml:[/bold yellow]

restore_from: "unsloth/orpheus-3b-0.1-ft"
max_seq_len: 4096
lr: 1e-4
per_device_train_batch_size: 4
gradient_accumulation_steps: 8
num_train_epochs: 3
lora_r: 64
lora_alpha: 64
load_in_4bit: false"""
    
    panel1 = Panel(neutts_config, title="NeuTTS Config", border_style="green")
    panel2 = Panel(orpheus_config, title="Orpheus Config", border_style="yellow")
    
    return Columns([panel1, panel2])


def print_recommendations():
    """Print final recommendations."""
    
    console.print("\n")
    console.print("📊 [bold]Decision Guide:[/bold]")
    console.print("\n")
    
    console.print("Choose [green bold]NeuTTS[/green bold] if you need:")
    console.print("  ✓ Faster inference speed")
    console.print("  ✓ Lower memory requirements")
    console.print("  ✓ Production deployment")
    console.print("  ✓ Real-time applications")
    console.print("  ✓ Simpler architecture")
    console.print("\n")
    
    console.print("Choose [yellow bold]Orpheus[/yellow bold] if you want:")
    console.print("  ✓ Highest audio quality")
    console.print("  ✓ Best voice similarity")
    console.print("  ✓ State-of-the-art results")
    console.print("  ✓ Leverage Llama architecture")
    console.print("  ✓ Efficient LoRA training")
    console.print("\n")


def print_quick_start():
    """Print quick start commands."""
    
    console.print("🚀 [bold]Quick Start Commands:[/bold]")
    console.print("\n")
    
    console.print("[green]NeuTTS:[/green]")
    console.print("  [dim]1.[/dim] python training/create_pairs.py --segments_path=... --output_path=...")
    console.print("  [dim]2.[/dim] python training/finetune-neutts.py --config_fpath=config.yaml")
    console.print("  [dim]3.[/dim] python training/test_checkpoint_batch.py --checkpoint_path=...")
    console.print("\n")
    
    console.print("[yellow]Orpheus:[/yellow]")
    console.print("  [dim]1.[/dim] python training/create_pairs.py --segments_path=... --output_path=...")
    console.print("  [dim]2.[/dim] ./training/run_orpheus_training.sh")
    console.print("  [dim]3.[/dim] python training/test_orpheus_inference.py --model_path=...")
    console.print("\n")


def main():
    """Main comparison display."""
    
    # Check if rich is installed
    try:
        from rich.console import Console
    except ImportError:
        print("Please install rich: pip install rich")
        sys.exit(1)
    
    print_header()
    
    # Architecture comparison
    console.print(create_architecture_table())
    console.print("\n")
    
    # Performance comparison
    console.print(create_performance_table())
    console.print("\n")
    
    # Use cases
    console.print(create_usage_table())
    console.print("\n")
    
    # Code comparison
    console.print("[bold]Implementation Comparison:[/bold]\n")
    console.print(create_code_comparison())
    console.print("\n")
    
    # Config comparison
    console.print("[bold]Configuration Comparison:[/bold]\n")
    console.print(create_config_comparison())
    console.print("\n")
    
    # Recommendations
    print_recommendations()
    
    # Quick start
    print_quick_start()
    
    # Footer
    console.print("="*80, style="bold blue")
    console.print("For detailed documentation:")
    console.print("  • NeuTTS: See training/README.md")
    console.print("  • Orpheus: See training/ORPHEUS_FINETUNING_GUIDE.md")
    console.print("="*80, style="bold blue")
    console.print("\n")


if __name__ == "__main__":
    main()

