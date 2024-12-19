from datetime import datetime, timezone
import traceback
import time
from collections import defaultdict
import requests
from rich.progress import track
from typing import Dict, Any
from .language_metrics import LanguageMetricsCollector
from .collaboration_metrics import CollaborationMetricsCollector


def _empty_metrics() -> Dict[str, Any]:
    return {
        'total_commits': 0,
        'lines_added': 0,
        'lines_removed': 0,
        'files_changed': 0,
        'commit_activity': {
            'hour_distribution': {},
            'day_distribution': {},
            'consistency_score': 0.0
        },
        'impact_score': 0
    }


def _calculate_consistency(commit_times: list) -> float:
    if not commit_times:
        return 0.0

    sorted_times = sorted(commit_times)

    # Calculate average time between commits
    time_diffs = [
        (sorted_times[i+1] - sorted_times[i]).total_seconds() / 3600  # Convert to hours
        for i in range(len(sorted_times)-1)
    ]

    if not time_diffs:
        return 0.0

    # Calculate consistency score based on regularity of commits
    avg_diff = sum(time_diffs) / len(time_diffs)
    variance = sum((diff - avg_diff) ** 2 for diff in time_diffs) / len(time_diffs)

    # Normalize to 0-1 scale (lower variance = higher consistency)
    consistency = 1 / (1 + variance/100)

    return consistency

def _calculate_impact_score(metrics: Dict[str, Any]) -> float:
    if not metrics.get('total_commits'):
        return 0.0

    additions = metrics.get('additions', 0)
    deletions = metrics.get('deletions', 0)
    files_changed = metrics.get('files_changed', 0)
    total_commits = metrics.get('total_commits', 0)
    total_lines = additions + deletions

    commit_score = min(100.0, (total_commits / 1000) * 100)
    lines_score = min(100.0, (total_lines / 100000) * 100)
    files_score = min(100.0, (files_changed / 5000) * 100)

    # Weighted average...
    impact_score = (
        commit_score * 0.35 +
        lines_score * 0.35 +
        files_score * 0.30
    )

    return round(impact_score, 2)

def _calculate_average_merge_time(prs: list) -> float:
    merge_times = []

    for index, pr in enumerate(prs):
        try:
            if (pr.get('state') == 'MERGED' and
                pr.get('createdAt') and
                pr.get('mergedAt')):

                created = datetime.fromisoformat(pr['createdAt'].replace('Z', '+00:00'))
                merged = datetime.fromisoformat(pr['mergedAt'].replace('Z', '+00:00'))
                merge_time = (merged - created).total_seconds() / 3600
                merge_times.append(merge_time)
            else:
                if index < 5:  # Show first few skipped
                    missing_fields = []
                    if pr.get('state') != 'MERGED': missing_fields.append('state != MERGED')
                    if not pr.get('createdAt'): missing_fields.append('createdAt')
                    if not pr.get('mergedAt'): missing_fields.append('mergedAt')
        except Exception as e:
            print(f"Error processing PR {index}: {str(e)}")
            continue

    if merge_times:
        average = sum(merge_times) / len(merge_times)
        return average
    else:
        return 0.0

def _process_commit_activity(commit_dates: list[datetime], contribution_weeks: list[Dict]) -> Dict[str, Any]:
    hour_distribution = defaultdict(int)
    day_distribution = defaultdict(int)

    for dt in commit_dates:
        hour_distribution[dt.hour] += 1
        day_distribution[dt.strftime('%A')] += 1

    for week in contribution_weeks:
        for day in week.get('contributionDays', []):
            count = day.get('contributionCount', 0)
            if count > 0:
                date = datetime.fromisoformat(day['date'])

                # Approximate hour distribution (spread across working hours)
                for hour in range(9, 18):  # 9 AM to 6 PM
                    hour_distribution[hour] += count / 9
                day_distribution[date.strftime('%A')] += count

    return {
        'hour_distribution': dict(hour_distribution),
        'day_distribution': dict(day_distribution),
        'consistency_score': _calculate_consistency(commit_dates)
    }

class GithubMetricsCollector:
    def __init__(self, token: str, username: str, year: int):
        self.token = token
        self.username = username
        self.year = year
        self.headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json"
        }
        self.api_base = "https://api.github.com"

        self.language_collector = LanguageMetricsCollector(self)
        self.collaboration_collector = CollaborationMetricsCollector(self)

    def graphql(self, query: str, variables: Dict[str, Any] = None) -> Dict[str, Any]:
        url = f"{self.api_base}/graphql"

        payload = {
            "query": query,
            "variables": variables or {}
        }

        response = requests.post(
            url,
            headers=self.headers,
            json=payload
        )

        if response.status_code != 200:
            raise Exception(f"GraphQL query failed: {response.text}")

        return response.json()

    def collect_all_metrics(self) -> Dict[str, Any]:
        metrics = {
            'code_changes': {
                'total_commits': 0,
                'lines_added': 0,
                'lines_removed': 0,
                'files_changed': 0,
                'commit_activity': {},
                'impact_score': 0,
            },
            'pull_requests': {
                'total_prs': 0,
                'merged_prs': 0,
                'merge_time_avg': 0,
                'merge_success_rate': 0
            },
            'code_reviews': {
                'total_reviews': 0,
                'review_comments': 0,
                'avg_review_length': 0,
                'response_time_avg': 0
            },
            'languages': {
                'language_distribution': {},
                'primary_languages': [],
            },
            'collaboration': {
                'collaboration_network': {
                    'collaborators': [],
                    'review_relationships': {}
                },
                'cross_repo_impact': {
                    'repositories': [],
                    'impact_score': 0
                }
            }
        }

        collection_steps = [
            ("Pull Requests", self._collect_pr_metrics),
            ("Code Changes", self._collect_code_metrics),
            ("Code Reviews", self._collect_review_metrics),
            ("Languages", self._collect_language_metrics),
            ("Collaboration", self._collect_collaboration_metrics)
        ]

        for step_name, collector_func in track(collection_steps, description="Collecting metrics..."):
            try:
                key = step_name.lower().replace(" ", "_")
                result = collector_func()
                if isinstance(result, dict):
                    metrics[key].update(result)
            except Exception as e:
                print(f"Error collecting {step_name}: {str(e)}")

        return metrics

    def _collect_review_metrics(self) -> Dict[str, Any]:
        return self.collaboration_collector.collect_review_metrics()

    def _collect_language_metrics(self) -> Dict[str, Any]:
        return self.language_collector.collect_language_metrics()

    def _collect_collaboration_metrics(self) -> Dict[str, Any]:
        return self.collaboration_collector.collect_collaboration_metrics()

    def _graphql_query(self, query: str) -> Dict[str, Any]:
        response = requests.post(
            f"{self.api_base}/graphql",
            headers=self.headers,
            json={'query': query}
        )
        response.raise_for_status()
        return response.json()

    def _get_user_email(self) -> str:
        email_query = """
        query {
            viewer {
                email
            }
        }
        """
        result = self.graphql(email_query)
        return result.get('data', {}).get('viewer', {}).get('email', '')

    def _check_rate_limit(self):
        query = """
        query {
            rateLimit {
                remaining
                resetAt
            }
        }
        """
        result = self.graphql(query)
        remaining = result.get('data', {}).get('rateLimit', {}).get('remaining', 0)
        if remaining < 100:  # Arbitrary threshold... ran into periodic issues
            reset_at = result.get('data', {}).get('rateLimit', {}).get('resetAt')
            reset_time = datetime.fromisoformat(reset_at.replace('Z', '+00:00'))
            wait_time = (reset_time - datetime.now(timezone.utc)).total_seconds()
            if wait_time > 0:
                print(f"\nRate limit low ({remaining} remaining). Waiting {wait_time:.0f} seconds...")
                time.sleep(wait_time + 1)

    def _collect_pr_metrics(self) -> Dict[str, Any]:
        basic_query = """
        query($from: DateTime!) {
            viewer {
                login
                contributionsCollection(from: $from) {
                    totalCommitContributions
                    commitContributionsByRepository {
                        repository {
                            name
                            owner {
                                login
                            }
                            defaultBranchRef {
                                name
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

        pr_query = """
        query($username: String!, $after: String) {
            user(login: $username) {
                pullRequests(first: 100, after: $after, orderBy: {field: CREATED_AT, direction: DESC}) {
                    totalCount
                    pageInfo {
                        hasNextPage
                        endCursor
                    }
                    nodes {
                        state
                        createdAt
                        mergedAt
                        title
                    }
                }
            }
        }
        """

        try:
            date_str = f"{self.year}-01-01T00:00:00Z"

            basic_vars = {"from": date_str}
            basic_result = self.graphql(basic_query, basic_vars)

            if 'data' not in basic_result or 'viewer' not in basic_result['data']:
                return _empty_metrics()

            # Get all PR data with pagination
            all_prs = []
            has_next_page = True
            after_cursor = None
            prs_date_threshold = datetime.fromisoformat(date_str.replace('Z', '+00:00'))


            while has_next_page:
                pr_vars = {
                    "username": self.username,
                    "after": after_cursor
                }
                pr_result = self.graphql(pr_query, pr_vars)

                pr_data = pr_result.get('data', {}).get('user', {}).get('pullRequests', {})

                current_page_prs = [
                    pr for pr in pr_data.get('nodes', [])
                    if datetime.fromisoformat(pr['createdAt'].replace('Z', '+00:00')) >= prs_date_threshold
                ]

                all_prs.extend(current_page_prs)

                page_info = pr_data.get('pageInfo', {})
                has_next_page = page_info.get('hasNextPage', False)
                after_cursor = page_info.get('endCursor')

            total_prs = len(all_prs)
            merged_prs = sum(1 for pr in all_prs if pr.get('state') == 'MERGED')
            merge_time_avg = _calculate_average_merge_time(all_prs)

            metrics = {
                'total_prs': total_prs,
                'merged_prs': merged_prs,
                'merge_time_avg': merge_time_avg,
                'merge_success_rate': (merged_prs / total_prs * 100) if total_prs > 0 else 0.0
            }

            return metrics

        except Exception as e:
            print(f"Error collecting code metrics: {str(e)}")
            import traceback
            traceback.print_exc()
            return _empty_metrics()

    def _collect_repository_commits(self, owner: str, name: str, since: str) -> list[Dict]:
        commit_query = """
        query CommitHistory($owner: String!, $name: String!, $since: GitTimestamp!, $after: String) {
            repository(owner: $owner, name: $name) {
                viewerPermission
                isPrivate
                defaultBranchRef {
                    target {
                        ... on Commit {
                            history(first: 100, since: $since, after: $after) {
                                pageInfo {
                                    hasNextPage
                                    endCursor
                                }
                                totalCount
                                nodes {
                                    additions
                                    deletions
                                    changedFiles
                                    committedDate
                                    author {
                                        name
                                        user {
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
        """

        all_commits = []
        after_cursor = None
        max_pages = 100
        current_page = 0
        total_count = None
        start_time = time.time()
        max_duration = 300  # 5 minute timeout per repository

        while current_page < max_pages:
            current_page += 1

            if time.time() - start_time > max_duration:
                print(f"\nTimeout reached for {owner}/{name} after {current_page} pages")
                break

            variables = {
                "owner": owner,
                "name": name,
                "since": since,
                "after": after_cursor
            }

            try:
                result = self.graphql(commit_query, variables)

                if 'errors' in result:
                    print(f"GraphQL errors for {owner}/{name}:", result['errors'])
                    break

                repository = result.get('data', {}).get('repository')
                if not repository or not repository.get('defaultBranchRef'):
                    print(f"No repository data for {owner}/{name}")
                    break

                history = (repository.get('defaultBranchRef', {})
                        .get('target', {})
                        .get('history', {}))

                if not history or not history.get('nodes'):
                    print(f"No commit history for {owner}/{name}")
                    break

                # Store total count on first page
                if total_count is None:
                    total_count = history.get('totalCount', 0)

                commits = history.get('nodes', [])
                valid_commits_count_before = len(all_commits)

                valid_commits = [
                    commit for commit in commits
                    if commit is not None and
                    isinstance(commit, dict) and
                    isinstance(commit.get('author', {}), dict) and
                    isinstance(commit.get('author', {}).get('user', {}), dict) and
                    commit.get('author', {}).get('user', {}).get('login') == self.username
                ]

                all_commits.extend(valid_commits)

                page_info = history.get('pageInfo', {})
                if not page_info or not page_info.get('hasNextPage') or not page_info.get('endCursor'):
                    break

                new_cursor = page_info.get('endCursor')
                if new_cursor == after_cursor:
                    print(f"Pagination not advancing for {owner}/{name}")
                    break

                after_cursor = new_cursor

                # Only sleep if we're continuing to next page
                time.sleep(0.1)

            except Exception as e:
                print(f"Error collecting commits for {owner}/{name}: {str(e)}")
                traceback.print_exc()
                break

        if current_page >= max_pages:
            print(f"Warning: Hit page limit for {owner}/{name}")

        return all_commits


    def _collect_code_metrics(self) -> Dict[str, Any]:
        basic_query = """
        query($username: String!, $from: DateTime!) {
            user(login: $username) {
                login
                contributionsCollection(from: $from) {
                    totalCommitContributions
                    commitContributionsByRepository {
                        repository {
                            name
                            owner {
                                login
                            }
                            defaultBranchRef {
                                name
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

        try:
            date_str = f"{self.year}-01-01T00:00:00Z"
            basic_vars = {"from": date_str, "username": self.username}

            basic_result = self.graphql(basic_query, basic_vars)

            if 'errors' in basic_result:
                print("GraphQL errors in basic query:", basic_result['errors'])
                return _empty_metrics()

            if 'data' not in basic_result or 'user' not in basic_result['data']:
                print(f"No data or user found for username: {self.username}")
                return _empty_metrics()

            user_data = basic_result['data']['user']
            contributions = user_data['contributionsCollection']
            repo_list = contributions.get('commitContributionsByRepository', [])

            lines_added = 0
            lines_removed = 0
            files_changed = 0
            commit_dates = []
            repositories_processed = []

            for index, repo_contrib in enumerate(repo_list, 1):
                repo = repo_contrib.get('repository', {})
                repo_name = repo.get('name')
                repo_owner = repo.get('owner', {}).get('login')

                if not repo_name or not repo_owner:
                    continue

                repo_commits = self._collect_repository_commits(repo_owner, repo_name, date_str)

                repositories_processed.append({
                    'name': f"{repo_owner}/{repo_name}",
                    'commits': len(repo_commits)
                })

                # Process commits
                for commit in repo_commits:
                    lines_added += commit.get('additions', 0)
                    lines_removed += commit.get('deletions', 0)
                    files_changed += commit.get('changedFiles', 0)

                    if commit_date := commit.get('committedDate'):
                        try:
                            parsed_date = datetime.fromisoformat(commit_date.replace('Z', '+00:00'))
                            commit_dates.append(parsed_date)
                        except (ValueError, TypeError) as e:
                            print(f"Error parsing date {commit_date}: {e}")

            metrics = {
                'total_commits': len(commit_dates),
                'lines_added': lines_added,
                'lines_removed': lines_removed,
                'files_changed': files_changed,
                'repositories_processed': repositories_processed,
                'commit_activity': _process_commit_activity(
                    commit_dates,
                    contributions.get('contributionCalendar', {}).get('weeks', [])
                ),
                'impact_score': _calculate_impact_score({
                    'total_commits': len(commit_dates),
                    'additions': lines_added,
                    'deletions': lines_removed,
                    'files_changed': files_changed
                })
            }

            return metrics

        except Exception as e:
            print(f"Error in _collect_code_metrics: {str(e)}")
            traceback.print_exc()
            return _empty_metrics()
