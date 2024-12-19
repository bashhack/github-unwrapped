import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, Any

class ChartGenerator:
    def __init__(self):
        self.theme = {
            'background': '#0d1117',
            'text': '#ffffff',
            'accent': '#58a6ff',
            'secondary': '#8b949e'
        }

        self.layout_template = {
            'paper_bgcolor': self.theme['background'],
            'plot_bgcolor': self.theme['background'],
            'font': {'color': self.theme['text']},
        }

    def create_commit_patterns_chart(self, metrics: Dict[str, Any]) -> go.Figure:
        activity = metrics['code_changes']['commit_activity']

        # Create subplot with two charts side by side...
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Commits by Day of Week', 'Commits by Hour'),
            column_widths=[0.4, 0.6]
        )

        # One for the daily pattern...
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        day_values = [activity['day_distribution'].get(day, 0) for day in days_order]

        # ...and one for the hourly pattern
        hours = sorted(activity['hour_distribution'].keys())
        hour_values = [activity['hour_distribution'].get(hour, 0) for hour in hours]

        def format_hour(hour):
            if hour == 0:
                return "12 AM"
            elif hour < 12:
                return f"{hour} AM"
            elif hour == 12:
                return "12 PM"
            else:
                return f"{hour - 12} PM"

        # Add daily pattern
        fig.add_trace(
            go.Bar(
                x=days_order,
                y=day_values,
                name='Daily Pattern',
                marker_color=self.theme['accent'],
                hovertemplate='Day: %{x}<br>Commits: %{y}<extra></extra>'
            ),
            row=1, col=1
        )

        # Add hourly pattern
        fig.add_trace(
            go.Bar(
                x=[format_hour(hour) for hour in hours],
                y=hour_values,
                name='Hourly Pattern',
                marker_color=self.theme['accent'],
                hovertemplate='Time: %{x}<br>Commits: %{y}<extra></extra>'
            ),
            row=1, col=2
        )

        # Create a custom layout that merges template and specific settings
        layout = {
            'paper_bgcolor': self.theme['background'],
            'plot_bgcolor': self.theme['background'],
            'font': {'color': self.theme['text']},
            'title': {
                'text': 'Commit Patterns',
                'font': {'size': 24},
                'y': 0.95
            },
            'height': 500,
            'showlegend': False,
            'margin': {'t': 100, 'b': 50}
        }

        fig.update_layout(layout)

        # Update x-axes
        fig.update_xaxes(
            title_text="Day of Week",
            row=1, col=1,
            gridcolor=self.theme['secondary']
        )
        fig.update_xaxes(
            title_text="Time of Day",
            tickangle=-45,
            dtick=1,
            row=1, col=2,
            gridcolor=self.theme['secondary']
        )

        # Update y-axes
        fig.update_yaxes(
            title_text="Number of Commits",
            row=1, col=1,
            gridcolor=self.theme['secondary']
        )
        fig.update_yaxes(
            title_text="Number of Commits",
            row=1, col=2,
            gridcolor=self.theme['secondary']
        )

        return fig

    def create_language_distribution_chart(self, metrics: Dict[str, Any]) -> go.Figure:
        languages = metrics['languages']['language_distribution']

        labels = list(languages.keys())
        values = list(languages.values())

        fig = go.Figure(data=go.Pie(
            labels=labels,
            values=values,
            hole=0.4,
            marker=dict(colors=px.colors.qualitative.Set3)
        ))

        layout = {
            **self.layout_template,
            'title': {'text': 'Language Distribution', 'font': {'size': 24}}
        }

        fig.update_layout(layout)
        return fig

    def create_pr_statistics_chart(self, metrics: Dict[str, Any]) -> go.Figure:
        pr_metrics = metrics['pull_requests']

        categories = ['Opened', 'Merged', 'Reviewed']
        values = [
            pr_metrics['total_prs'],
            pr_metrics['merged_prs'],
            metrics['code_reviews']['total_reviews']
        ]

        fig = go.Figure(data=go.Bar(
            x=categories,
            y=values,
            marker_color=self.theme['accent']
        ))

        layout = {
            **self.layout_template,
            'title': {'text': 'Pull Request Activity', 'font': {'size': 24}},
            'xaxis_title': 'Category',
            'yaxis_title': 'Count'
        }

        fig.update_layout(layout)
        return fig

    def create_review_impact_chart(self, metrics: Dict[str, Any]) -> go.Figure:
        reviews = metrics['code_reviews']
        review_patterns = reviews.get('review_patterns', {})

        repo_data = review_patterns.get('by_repository', {})

        if not repo_data:
            fig = go.Figure()
            layout = {
                **self.layout_template,
                'title': {'text': 'Code Review Distribution by Repository', 'font': {'size': 24}},
                'xaxis_title': 'Repository',
                'yaxis_title': 'Number of Reviews',
                'annotations': [{
                    'text': 'No review data available',
                    'xref': 'paper',
                    'yref': 'paper',
                    'showarrow': False,
                    'font': {'size': 16, 'color': self.theme['secondary']},
                    'x': 0.5,
                    'y': 0.5
                }]
            }
            fig.update_layout(layout)
            return fig

        # Sort repositories by number of reviews
        sorted_repos = sorted(
            repo_data.items(),
            key=lambda x: x[1],
            reverse=True
        )

        repos = [repo for repo, _ in sorted_repos]
        review_counts = [count for _, count in sorted_repos]

        fig = go.Figure(data=go.Bar(
            x=repos,
            y=review_counts,
            marker_color=self.theme['accent'],
            text=review_counts,
            textposition='auto',
        ))

        layout = {
            **self.layout_template,
            'title': {'text': 'Code Review Distribution by Repository', 'font': {'size': 24}},
            'xaxis_title': 'Repository',
            'yaxis_title': 'Number of Reviews',
            'xaxis': {
                'tickangle': -45,
                'automargin': True
            },
            'bargap': 0.2,
            'height': 600  # Make chart taller to accommodate repository names (probably a better way to do this lol)
        }

        fig.update_layout(layout)
        return fig

    def generate_charts(self, metrics: Dict[str, Any]) -> Dict[str, go.Figure]:
        return {
            'commit_activity': self.create_commit_patterns_chart(metrics),
            'language_distribution': self.create_language_distribution_chart(metrics),
            'pr_statistics': self.create_pr_statistics_chart(metrics),
            'review_impact': self.create_review_impact_chart(metrics)
        }
