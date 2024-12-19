import click
from rich.console import Console
from pathlib import Path
import json
from datetime import datetime
import os
from dotenv import load_dotenv

from collectors.metrics_collector import GithubMetricsCollector
from analyzers.insights_generator import InsightsGenerator
from visualizers.chart_generator import ChartGenerator

load_dotenv()

console = Console()

class GithubUnWrapped:
    def __init__(self, token, username, year, output_dir="output"):
        self.token = token
        self.username = username
        self.year = year
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_collector = GithubMetricsCollector(token, username, year)
        self.insights_generator = InsightsGenerator()
        self.chart_generator = ChartGenerator()

    def generate(self):
        try:
            # 1. Collect Metrics
            with console.status("[bold green]Collecting your Github metrics...") as status:
                metrics = self.metrics_collector.collect_all_metrics()

                metrics["username"] = self.username

                status.update("[bold green]Analyzing your code journey...")
                insights = self.insights_generator.generate_insights(metrics)

            # 2. Generate Visualizations
            console.print("\n[bold cyan]Creating your analysis...[/bold cyan]")
            charts = self.chart_generator.generate_charts(metrics)

            # 3. Save Results
            self.save_results(metrics, insights, charts)

            # 4. Present Summary
            self.present_unwrapped(metrics, insights)

            return metrics, insights

        except Exception as e:
            console.print(f"[bold red]Error generating Github UnWrapped: {str(e)}[/bold red]")
            raise

    def save_results(self, metrics, insights, charts):
        report = {
            'metrics': metrics,
            'insights': insights,
            'generated_at': datetime.now().isoformat(),
            'username': self.username,
            'year': self.year
        }

        report_path = self.output_dir / f"github_unwrapped_{self.username}_{self.year}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        charts_dir = self.output_dir / "charts"
        charts_dir.mkdir(exist_ok=True)
        for name, chart in charts.items():
            chart.write_image(charts_dir / f"{name}.png")
            chart.write_html(charts_dir / f"{name}.html")

    def present_unwrapped(self, metrics: dict[str, any], insights: dict[str, any]):
        console.print("\n" + "=" * 50)
        console.print(f"[bold magenta]🎉 Your Github Un-Wrapped {self.year} 🎉[/bold magenta]")
        console.print("=" * 50)

        console.print("\n[bold cyan]📊 Impact Summary[/bold cyan]")
        console.print("=" * 20)

        code_stars = "⭐ " * 5 if metrics['code_changes']['total_commits'] >= 500 else "⭐ " * 4
        review_stars = "⭐ " * 5 if metrics['code_reviews']['total_reviews'] >= 200 else "⭐ " * 4
        quality_stars = "⭐ " * 5 if metrics['pull_requests']['merge_success_rate'] >= 90 else "⭐ " * 4

        console.print(f"Code:     {code_stars}")
        console.print(f"Reviews:  {review_stars}")
        console.print(f"Quality:  {quality_stars}")

        console.print("\n[bold cyan]🎯 Code Impact[/bold cyan]")
        console.print("-" * 20)
        console.print(f"Total Commits: [bold]{metrics['code_changes']['total_commits']:,}[/bold]")
        console.print(f"Lines Added: [bold]{metrics['code_changes']['lines_added']:,}[/bold]")
        console.print(f"Lines Removed: [bold]{metrics['code_changes']['lines_removed']:,}[/bold]")
        console.print(f"Files Changed: [bold]{metrics['code_changes']['files_changed']:,}[/bold]")

        console.print("\n[bold cyan]🔄 Pull Requests[/bold cyan]")
        console.print("-" * 20)
        success_rate = metrics['pull_requests']['merge_success_rate']
        merge_indicator = "🟢" if success_rate >= 90 else "🟡" if success_rate >= 80 else "🔴"

        console.print(f"Total PRs: [bold]{metrics['pull_requests']['total_prs']}[/bold]")
        console.print(f"Merged PRs: [bold]{metrics['pull_requests']['merged_prs']}[/bold]")
        console.print(f"Average Merge Time: [bold]{metrics['pull_requests']['merge_time_avg']:.1f}[/bold] hours")
        console.print(f"Success Rate: [bold]{success_rate:.1f}%[/bold] {merge_indicator}")

        console.print("\n[bold cyan]👩‍💻 Top Languages[/bold cyan]")
        console.print("-" * 20)
        for lang, percentage in metrics['languages']['primary_languages'][:5]:
            bar_length = int(percentage / 2)  # Scale percentage to reasonable bar length
            bar = "█" * bar_length
            console.print(f"{lang:<12} [bold]{percentage:>5.1f}%[/bold] {bar}")

        console.print("\n[bold cyan]📝 Code Reviews[/bold cyan]")
        console.print("-" * 20)
        console.print(f"Total Reviews: [bold]{metrics['code_reviews']['total_reviews']}[/bold]")
        console.print(
            f"Repositories Reviewed: [bold]{metrics['code_reviews']['cross_repo_reviews']['unique_repositories']}[/bold]")
        console.print(
            f"Review Engagement: [bold]{metrics['code_reviews']['review_patterns']['engagement_level']}[/bold]")

        if 'achievements' in insights:
            achievements_by_level = {
                'platinum': {'achievements': [], 'symbol': '🏆'},
                'gold': {'achievements': [], 'symbol': '🥇'},
            }

            for achievement in insights['achievements']:
                level = achievement['level']
                if level in achievements_by_level:
                    achievements_by_level[level]['achievements'].append(achievement)

            for level, data in achievements_by_level.items():
                if data['achievements']:
                    console.print(f"\n[bold cyan]{data['symbol']} {level.title()} Achievements[/bold cyan]")
                    console.print("-" * 50)
                    for ach in data['achievements']:
                        console.print(f"{ach['icon']} {ach['title']}: {ach['description']}")

        if 'summary_insights' in insights and 'strength_areas' in insights['summary_insights']:
            console.print("\n[bold cyan]💪 Strength Areas[/bold cyan]")
            console.print("-" * 50)
            for i, strength in enumerate(insights['summary_insights']['strength_areas'], 1):
                console.print(f"{i}: {strength['area']}")
                console.print(f"   Evidence: [bold]{strength['evidence']}[/bold]")
                console.print(f"   Level: [bold]{strength['level']}[/bold]")

        if 'recommendations' in insights:
            console.print("\n[bold cyan]📈 Growth Opportunities[/bold cyan]")
            console.print("-" * 50)
            for i, rec in enumerate(insights['recommendations'][:3], 1):
                console.print(f"{i}: [bold]{rec['title']}[/bold]")
                console.print(f"   {rec['description']}")
                if 'metrics' in rec:
                    # Safely access metrics with more flexible structure
                    metrics_info = rec['metrics']
                    if isinstance(metrics_info, dict):
                        for key, value in metrics_info.items():
                            # Format key to be more readable
                            formatted_key = key.replace('_', ' ').title()
                            console.print(f"   {formatted_key}: {value}")

        # Output location (JSON+ charts (HTML + PNG) ...)
        console.print(f"\n[bold green]Complete report saved to: {self.output_dir}[/bold green]")

@click.command()
@click.option('--token', '-t', default=lambda: os.getenv('GITHUB_TOKEN'),
              help='Github Personal Access Token')
@click.option('--username', '-u', required=True, help='Github username')
@click.option('--year', '-y', default=datetime.now().year, help='Year to analyze')
@click.option('--output', '-o', default='output', help='Output directory')
def cli(token, username, year, output):
    if not token:
        console.print("[bold red]Error: Github token is required. "
                     "Either provide it via --token or set GITHUB_TOKEN "
                     "environment variable.[/bold red]")
        return

    unwrapped = GithubUnWrapped(token, username, year, output)
    unwrapped.generate()

if __name__ == '__main__':
    cli()
