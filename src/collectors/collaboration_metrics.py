import time
from typing import Dict, Any, List
import traceback
from datetime import datetime
from collections import defaultdict, Counter


def _empty_review_metrics() -> Dict[str, Any]:
    return {
        'total_reviews': 0,
        'review_patterns': {
            'by_repository': {},
            'by_author': {},
            'engagement_level': 'Low'
        },
        'cross_repo_reviews': {
            'unique_repositories': 0,
            'unique_authors': 0,
            'review_spread': 0
        },
        'impact_metrics': {
            'total_contributions': 0,
            'review_ratio': 0
        }
    }

def _calculate_engagement_level(total_reviews: int) -> str:
    if total_reviews >= 100:
        return "Very High"
    elif total_reviews >= 50:
        return "High"
    elif total_reviews >= 20:
        return "Moderate"
    elif total_reviews > 0:
        return "Low"
    return "None"

def _calculate_review_efficiency(cycles: Dict[int, int], avg_cycle_time: float) -> float:
    # Calculate review efficiency score (0-10)...

    # Factors that indicate efficiency:
    # - High percentage of single-review approvals
    # - Low average cycle time
    # - Few very high cycle counts

    total_reviews = sum(cycles.values())
    if not total_reviews:
        return 0.0

    single_review_ratio = cycles.get(1, 0) / total_reviews
    high_cycle_ratio = sum(count for cycle, count in cycles.items() if cycle > 5) / total_reviews

    # Score components (0-1 scale)
    single_review_score = single_review_ratio
    cycle_time_score = 1.0 / (1.0 + avg_cycle_time / 24.0)  # Normalize by day
    high_cycle_penalty = 1.0 - high_cycle_ratio

    # Combine scores with weights
    weighted_score = (
            single_review_score * 0.4 +
            cycle_time_score * 0.4 +
            high_cycle_penalty * 0.2
    )

    return min(10.0, weighted_score * 10)

def _analyze_review_cycles(cycles: List[int], pr_stats: Dict) -> Dict[str, Any]:
    if not cycles:
        return {
            'average_cycles': 0,
            'cycle_distribution': {},
            'complex_reviews': [],
            'efficiency_score': 0
        }

    cycle_distribution = Counter(cycles)
    total_prs = len(cycles)

    # Find complex reviews (those with higher than average review cycles)
    avg_cycles = sum(cycles) / total_prs
    threshold = max(2, avg_cycles + 1)

    complex_reviews = [
        {
            'pr': pr_key,
            'cycles': stats['cycle_count'],
            'author': stats['author']
        }
        for pr_key, stats in pr_stats.items()
        if stats.get('cycle_count', 0) >= threshold
    ]

    complex_reviews.sort(key=lambda x: x['cycles'], reverse=True)

    return {
        'average_cycles': avg_cycles,
        'cycle_distribution': dict(cycle_distribution),
        'single_review_percentage': (cycle_distribution.get(1, 0) / total_prs * 100),
        'multiple_review_percentage': (sum(v for k, v in cycle_distribution.items() if k > 1) / total_prs * 100),
        'complex_reviews': complex_reviews[:5],  # Top 5 most complex reviews
        'efficiency_score': _calculate_review_efficiency(cycle_distribution, avg_cycles)
    }

class CollaborationMetricsCollector:
    def __init__(self, api_client):
        self.api = api_client

    def _print_dict_structure(self, d: Dict, indent: int = 0):
        for key, value in d.items():
            if isinstance(value, dict):
                print("  " * indent + f"{key}:")
                self._print_dict_structure(value, indent + 1)
            elif isinstance(value, list):
                print("  " * indent + f"{key}: [{len(value)} items]")
            else:
                print("  " * indent + f"{key}: {value}")

    def collect_collaboration_metrics(self) -> Dict[str, Any]:
        def _ensure_json_serializable(obj):
            if isinstance(obj, set):
                return list(obj)
            elif isinstance(obj, dict):
                return {k: _ensure_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [_ensure_json_serializable(item) for item in obj]
            elif isinstance(obj, (str, int, float, bool, type(None))):
                return obj
            else:
                return str(obj)

        try:
            def get_repos_query(after_cursor=None):
                after_arg = f', after: "{after_cursor}"' if after_cursor else ''
                return """
                query($username: String!) {
                    user(login: $username) {
                        repositoriesContributedTo(
                            first: 25,
                            contributionTypes: [COMMIT, ISSUE, PULL_REQUEST, PULL_REQUEST_REVIEW]
                            %s
                        ) {
                            pageInfo {
                                hasNextPage
                                endCursor
                            }
                            nodes {
                                name
                                owner {
                                    login
                                }
                                viewerPermission
                                isPrivate
                                primaryLanguage {
                                    name
                                }
                                pullRequests(
                                    first: 20, 
                                    states: [OPEN, MERGED],
                                    orderBy: {field: CREATED_AT, direction: DESC}
                                ) {
                                    pageInfo {
                                        hasNextPage
                                        endCursor
                                    }
                                    nodes {
                                        number
                                        createdAt
                                        author {
                                            login
                                        }
                                        reviews(first: 10) {
                                            nodes {
                                                author {
                                                    login
                                                }
                                                state
                                                createdAt
                                            }
                                        }
                                        reviewRequests(first: 10) {
                                            nodes {
                                                requestedReviewer {
                                                    ... on User {
                                                        login
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                """ % after_arg

            variables = {
                "username": self.api.username
            }

            metrics = {
                'collaboration_network': {
                    'collaborators': [],
                    'review_relationships': {},
                    'contribution_repos': []
                },
                'cross_repo_impact': {
                    'repositories': [],
                    'impact_score': 0,
                    'review_activity': {}
                }
            }

            # For collecting data before final conversion...
            temp_data = {
                'collaborators': set(),
                'contribution_repos': set(),
                'repositories': set(),
                'review_relationships': defaultdict(int),
                'review_activity': defaultdict(lambda: {
                    'reviews_given': 0,
                    'reviews_received': 0,
                    'permission_level': 'NONE',
                    'is_private': False,
                    'primary_language': None
                })
            }

            has_next_page = True
            after_cursor = None
            total_repos_processed = 0

            while has_next_page:
                query = get_repos_query(after_cursor)
                data = self.api.graphql(query, variables)

                if 'errors' in data:
                    break

                if 'data' not in data or 'user' not in data['data']:
                    print("No data returned from API")
                    break

                repos_data = data['data']['user'].get('repositoriesContributedTo', {})
                if not repos_data:
                    print("No repository data found")
                    break

                repos = repos_data.get('nodes', [])
                if not repos:
                    print("No repositories found in nodes")
                    break

                total_repos_processed += len(repos)

                for repo in repos:
                    if not repo:
                        continue

                    try:
                        repo_owner = repo.get('owner', {}).get('login')
                        repo_name = repo.get('name')

                        if not repo_owner or not repo_name:
                            print(f"Skipping repository with missing owner or name")
                            continue

                        full_repo_name = f"{repo_owner}/{repo_name}"

                        temp_data['repositories'].add(full_repo_name)

                        repo_metadata = {
                            'permission_level': repo.get('viewerPermission', 'NONE'),
                            'is_private': repo.get('isPrivate', False),
                            'primary_language': None,
                            'reviews_given': 0,
                            'reviews_received': 0
                        }

                        if repo.get('primaryLanguage'):
                            repo_metadata['primary_language'] = repo['primaryLanguage'].get('name')

                        temp_data['review_activity'][full_repo_name].update(repo_metadata)

                        # Calculate impact score based on permissions
                        permission_scores = {
                            'ADMIN': 5,
                            'MAINTAIN': 4,
                            'WRITE': 3,
                            'TRIAGE': 2,
                            'READ': 1,
                            'NONE': 0
                        }
                        permission_level = repo.get('viewerPermission', 'NONE')
                        metrics['cross_repo_impact']['impact_score'] += permission_scores.get(permission_level, 0)

                        pull_requests = repo.get('pullRequests', {}).get('nodes', []) or []
                        for pr in pull_requests:
                            if not pr:
                                continue

                            try:
                                pr_created = datetime.fromisoformat(pr['createdAt'].replace('Z', '+00:00'))
                                if pr_created.year != self.api.year:
                                    continue
                            except (ValueError, KeyError) as e:
                                print(f"Error parsing PR date in {full_repo_name}: {e}")
                                continue

                            pr_author = pr.get('author', {}).get('login')
                            if not pr_author:
                                continue

                            pr_number = pr.get('number')
                            if not pr_number:
                                continue

                            # Track PR author interactions
                            if pr_author != self.api.username:
                                temp_data['collaborators'].add(pr_author)

                            reviews = pr.get('reviews', {}).get('nodes', []) or []
                            for review in reviews:
                                if not review:
                                    continue

                                review_author = review.get('author', {}).get('login')
                                if not review_author:
                                    continue

                                if review_author == self.api.username:
                                    temp_data['review_activity'][full_repo_name]['reviews_given'] += 1
                                    temp_data['review_relationships'][pr_author] += 1
                                elif pr_author == self.api.username:
                                    temp_data['review_activity'][full_repo_name]['reviews_received'] += 1
                                    temp_data['collaborators'].add(review_author)

                            review_requests = pr.get('reviewRequests', {}).get('nodes', []) or []
                            for request in review_requests:
                                if not request:
                                    continue

                                reviewer = request.get('requestedReviewer', {}).get('login')
                                if reviewer:
                                    temp_data['collaborators'].add(reviewer)

                    except Exception as repo_error:
                        print(f"Error processing repository: {repo_error}")
                        continue

                page_info = repos_data.get('pageInfo', {})
                has_next_page = page_info.get('hasNextPage', False)
                after_cursor = page_info.get('endCursor')

                time.sleep(0.1)

            # Convert temporary data to final metrics
            metrics['collaboration_network']['collaborators'] = list(temp_data['collaborators'])
            metrics['collaboration_network']['contribution_repos'] = list(temp_data['contribution_repos'])
            metrics['collaboration_network']['review_relationships'] = dict(temp_data['review_relationships'])
            metrics['cross_repo_impact']['repositories'] = list(temp_data['repositories'])
            metrics['cross_repo_impact']['review_activity'] = dict(temp_data['review_activity'])

            try:
                total_reviews = sum(
                    activity['reviews_given']
                    for activity in metrics['cross_repo_impact']['review_activity'].values()
                )

                if total_reviews > 0 and total_repos_processed > 0:
                    metrics['cross_repo_impact']['impact_score'] = min(
                        100,
                        (metrics['cross_repo_impact']['impact_score'] / total_repos_processed) * 20
                    )
            except Exception as calc_error:
                print(f"Error calculating final metrics: {calc_error}")

            final_metrics = _ensure_json_serializable(metrics)
            return final_metrics

        except Exception as e:
            print(f"Error in collaboration metrics collection: {str(e)}")
            traceback.print_exc()
            return {
                'collaboration_network': {
                    'collaborators': [],
                    'review_relationships': {},
                    'contribution_repos': []
                },
                'cross_repo_impact': {
                    'repositories': [],
                    'impact_score': 0,
                    'review_activity': {}
                }
            }

    def collect_review_metrics(self) -> Dict[str, Any]:
        try:
            def get_review_query(after_cursor=None):
                after_arg = f', after: "{after_cursor}"' if after_cursor else ''
                return f"""
                query($username: String!, $from: DateTime!) {{
                    user(login: $username) {{
                        contributionsCollection(from: $from) {{
                            totalPullRequestReviewContributions
                            pullRequestReviewContributions(first: 50{after_arg}) {{
                                nodes {{
                                    occurredAt
                                    pullRequest {{
                                        createdAt
                                        number
                                        title
                                        author {{
                                            login
                                        }}
                                        repository {{
                                            name
                                            owner {{
                                                login
                                            }}
                                        }}
                                        reviews(first: 20) {{
                                            totalCount
                                            nodes {{
                                                author {{
                                                    login
                                                }}
                                                body
                                                createdAt
                                                state
                                                submittedAt
                                                comments {{
                                                    totalCount
                                                }}
                                            }}
                                        }}
                                        commits(last: 1) {{
                                            totalCount
                                        }}
                                    }}
                                }}
                                pageInfo {{
                                    hasNextPage
                                    endCursor
                                }}
                            }}
                        }}
                    }}
                }}
                """

            variables = {
                "username": self.api.username,
                "from": f"{self.api.year}-01-01T00:00:00Z"
            }

            all_review_nodes = []
            has_next_page = True
            after_cursor = None
            total_reviews = None

            while has_next_page:
                query = get_review_query(after_cursor)
                response = self.api.graphql(query, variables)

                if 'errors' in response:
                    break

                if 'data' not in response or 'user' not in response['data']:
                    break

                contributions = response['data']['user']['contributionsCollection']

                if total_reviews is None:
                    total_reviews = contributions['totalPullRequestReviewContributions']

                review_data = contributions['pullRequestReviewContributions']
                current_nodes = review_data['nodes']

                filtered_nodes = []
                for node in current_nodes:
                    if node and 'occurredAt' in node:
                        occurred_at = datetime.fromisoformat(node['occurredAt'].replace('Z', '+00:00'))
                        if occurred_at.year == self.api.year:
                            filtered_nodes.append(node)

                all_review_nodes.extend(filtered_nodes)

                page_info = review_data['pageInfo']
                has_next_page = page_info['hasNextPage']
                after_cursor = page_info['endCursor']

            total_comments = 0
            total_review_length = 0
            response_times = []
            review_cycles = defaultdict(int)
            pr_stats = defaultdict(lambda: {
                'changes_requested': 0,
                'last_state': None,
                'last_review_time': None,
                'review_dates': [],
                'author': None
            })

            review_stats = {
                'by_repository': defaultdict(int),
                'by_author': defaultdict(int),
                'pr_authors': set(),
                'total_reviews': len(all_review_nodes)
            }

            for node in all_review_nodes:
                if not node or 'pullRequest' not in node:
                    continue

                pr = node['pullRequest']
                repo_owner = pr['repository']['owner']['login']
                repo_name = pr['repository']['name']
                full_repo_name = f"{repo_owner}/{repo_name}"
                pr_key = f"{full_repo_name}#{pr['number']}"
                pr_author = pr['author']['login']

                review_stats['by_repository'][full_repo_name] += 1
                review_stats['pr_authors'].add(pr_author)

                reviews = sorted(
                    [r for r in pr['reviews']['nodes'] if r and r.get('state') in ['CHANGES_REQUESTED', 'APPROVED']],
                    key=lambda x: x.get('createdAt', '')
                )

                # Track meaningful review cycles
                cycle_count = 0
                last_review_state = None

                for review in reviews:
                    review_author = review['author']['login']
                    review_state = review['state']

                    try:
                        pr_created = datetime.fromisoformat(review['createdAt'].replace('Z', '+00:00'))
                    except (ValueError, KeyError):
                        return [], [], 0

                    if review_author == self.api.username:
                        total_comments += review['comments']['totalCount']
                        if review['body']:
                            total_review_length += len(review['body'])

                        if review.get('submittedAt'):
                            submitted = datetime.fromisoformat(review['submittedAt'].replace('Z', '+00:00'))
                            response_time = (submitted - pr_created).total_seconds() / 3600
                            if response_time >= 0:  # Only append valid response times
                                response_times.append(response_time)

                        # Determine if this is a new cycle
                        is_new_cycle = False
                        if review_state == 'CHANGES_REQUESTED':
                            is_new_cycle = True
                        elif review_state == 'APPROVED':
                            # Only count as a cycle if it's the first review or follows changes
                            is_new_cycle = (last_review_state is None or
                                            last_review_state == 'CHANGES_REQUESTED')

                        if is_new_cycle:
                            cycle_count += 1

                        last_review_state = review_state

                if cycle_count > 0:
                    review_cycles[pr_key] = cycle_count
                    pr_stats[pr_key].update({
                        'cycle_count': cycle_count,
                        'author': pr_author,
                        'title': pr['title']
                    })

            cycle_analysis = _analyze_review_cycles(list(review_cycles.values()), pr_stats)

            review_metrics = {
                'total_reviews': len(all_review_nodes),
                'review_comments': total_comments,
                'avg_review_length': total_review_length / len(all_review_nodes) if all_review_nodes else 0,
                'response_time_avg': sum(response_times) / len(response_times) if response_times else 0,
                'review_patterns': {
                    'by_repository': dict(review_stats['by_repository']),
                    'engagement_level': _calculate_engagement_level(len(all_review_nodes))
                },
                'cross_repo_reviews': {
                    'unique_repositories': len(review_stats['by_repository']),
                    'unique_authors': len(review_stats['pr_authors']),
                    'review_spread': len(review_stats['by_repository']) / len(all_review_nodes) if all_review_nodes else 0
                },
                'review_cycles': list(review_cycles.values()),
                'cycle_analysis': cycle_analysis
            }

            return review_metrics

        except Exception as e:
            print(f"\nError collecting review metrics: {str(e)}")
            import traceback
            traceback.print_exc()
            return _empty_review_metrics()
