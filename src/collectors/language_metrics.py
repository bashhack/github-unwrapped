from typing import Dict, Any
from collections import defaultdict


class LanguageMetricsCollector:
    def __init__(self, api_client):
        self.api = api_client

    def collect_language_metrics(self) -> Dict[str, Any]:
        try:
            query = """
            query($username: String!, $from: DateTime!) {
                user(login: $username) {
                    contributionsCollection(from: $from) {
                        commitContributionsByRepository {
                            repository {
                                name
                                languages(first: 10, orderBy: {field: SIZE, direction: DESC}) {
                                    totalSize
                                    edges {
                                        size
                                        node {
                                            name
                                        }
                                    }
                                }
                            }
                            contributions {
                                totalCount
                            }
                        }
                    }
                }
            }
            """

            variables = {
                "username": self.api.username,
                "from": f"{self.api.year}-01-01T00:00:00Z"
            }

            response = self.api.graphql(query, variables)

            metrics = {
                'language_distribution': {},
                'primary_languages': [],
            }

            if 'data' not in response or 'user' not in response['data']:
                print("Debug: No valid data in response")
                return metrics

            contributions = response['data']['user']['contributionsCollection']
            repo_contributions = contributions.get('commitContributionsByRepository', [])

            # Track languages weighted by contributions
            language_weights = defaultdict(float)
            total_weight = 0

            for repo_contrib in repo_contributions:
                repo = repo_contrib['repository']
                contribution_count = repo_contrib['contributions']['totalCount']

                if contribution_count > 0 and repo.get('languages'):
                    languages = repo['languages']['edges']
                    repo_total_size = repo['languages']['totalSize']

                    if repo_total_size > 0:
                        for lang_edge in languages:
                            lang_name = lang_edge['node']['name']
                            lang_size = lang_edge['size']

                            # Weight language by both size and contribution count
                            weight = (lang_size / repo_total_size) * contribution_count
                            language_weights[lang_name] += weight
                            total_weight += weight

            if total_weight > 0:

                metrics['language_distribution'] = {
                    lang: (weight / total_weight) * 100
                    for lang, weight in language_weights.items()
                }

                # Sort languages by percentage and filter out minimal contributions
                metrics['primary_languages'] = sorted(
                    [(lang, pct) for lang, pct in metrics['language_distribution'].items() if pct >= 1.0],
                    key=lambda x: x[1],
                    reverse=True
                )

            return metrics

        except Exception as e:
            print(f"\nError in language metrics collection: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'language_distribution': {},
                'primary_languages': [],
            }
