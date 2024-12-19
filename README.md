# Github Un-Wrapped

Generate data-driven insights about your Github contributions, similar to Spotify Wrapped but for your code!

## Features

- 📊 Comprehensive metrics analysis
  - Code changes and commit patterns
  - Pull request statistics
  - Code review impact
  - Language distribution
  - Collaboration patterns

- 🎨 Visualizations
  - Commit pattern heatmaps
  - Language distribution charts
  - PR statistics
  - Review impact analysis

- 🏆 Achievement System
  - Recognizes various levels of contributions
  - Platinum and Gold tier achievements
  - Custom badges for different accomplishments

- 📈 Growth Analytics
  - Identifies strength areas
  - Suggests improvement opportunities
  - Tracks progress across different metrics

## Installation

1. Clone the repository:
```bash
git clone https://github.com/bashhack/github-unwrapped.git
cd github-unwrapped
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up your Github token:
   - Create a [Personal Access Token](https://github.com/settings/tokens) with `repo` and `read:user` scopes
   - Either set it as an environment variable:
     ```bash
     export GITHUB_TOKEN=your_token_here
     ```
   - Or create a `.env` file:
     ```
     GITHUB_TOKEN=your_token_here
     ```

## Usage

```bash
python src/main.py --username <github_username> [--year 2024] [--output output_dir]
```

Options:
- `--username`, `-u`: Your Github username (required)
- `--year`, `-y`: Year to analyze (defaults to current year)
- `--output`, `-o`: Output directory (defaults to 'output')
- `--token`, `-t`: Github token (optional if set in env or .env)

## Output

The tool generates:
1. A comprehensive JSON report
2. Interactive HTML visualizations
3. Static PNG charts
4. Console-based summary with:
   - Impact metrics
   - Achievement badges
   - Strength areas
   - Growth opportunities

## Architecture

- `main.py`: Entry point and orchestration
- `collectors/`: Data collection from Github API
  - `metrics_collector.py`: Core metrics collection
  - `language_metrics.py`: Language analysis
  - `collaboration_metrics.py`: Team collaboration metrics
- `analyzers/`: Data analysis and insights generation
  - `insights_generator.py`: Generates insights from metrics
- `visualizers/`: Data visualization
  - `chart_generator.py`: Creates charts and visualizations

## Requirements

- Python 3.10+
- Github Personal Access Token (Classic PAT)
- Required packages listed in `requirements.txt`

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Inspired by Spotify Wrapped
- Built with Plotly for visualizations
- Uses Github's GraphQL API for data collection
