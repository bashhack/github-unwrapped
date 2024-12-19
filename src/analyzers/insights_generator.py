from typing import Dict, Any, List
import numpy as np


def _get_percentile_rank(value: float, benchmarks: Dict[int, float]) -> int:
    for percentile in sorted(benchmarks.keys(), reverse=True):
        if value >= benchmarks[percentile]:
            return percentile
    return 25  # Default to 25th percentile if below all benchmarks


def _analyze_code_patterns(metrics: Dict[str, Any]) -> Dict[str, Any]:
    code_changes = metrics['code_changes']

    # Calculate actual averages
    total_commits = code_changes['total_commits']
    files_per_commit = code_changes['files_changed'] / total_commits if total_commits > 0 else 0

    total_lines_changed = code_changes['lines_added'] + code_changes['lines_removed']
    avg_change_size = total_lines_changed / total_commits if total_commits > 0 else 0

    # Get actual peak day and time from distributions
    day_distribution = code_changes['commit_activity']['day_distribution']
    hour_distribution = code_changes['commit_activity']['hour_distribution']

    peak_day = max(day_distribution.items(), key=lambda x: x[1])[0]
    peak_hour = max(hour_distribution.items(), key=lambda x: x[1])[0]
    peak_time = f"{peak_hour:02d}:00"

    # Calculate daily and weekly averages
    daily_avg = total_commits / 365  # Assume yearly data...
    weekly_avg = daily_avg * 7

    return {
        'commit_patterns': {
            'average_commits_per_day': round(daily_avg, 2),
            'average_commits_per_week': round(weekly_avg, 2),
            'peak_day': peak_day,
            'peak_time': peak_time
        },
        'code_size_patterns': {
            'average_change_size': round(avg_change_size, 2),
            'files_per_commit': round(files_per_commit, 2)
        },
        'impact_patterns': {
            'average_impact_per_commit': round(code_changes['impact_score'] / total_commits, 2),
            'lines_changed_per_commit': round(total_lines_changed / total_commits, 2)
        }
    }


def _analyze_workflow_patterns(metrics: Dict[str, Any]) -> Dict[str, Any]:
    code_reviews = metrics.get('code_reviews', {})
    prs = metrics.get('pull_requests', {})
    code_changes = metrics.get('code_changes', {})
    cycle_analysis = code_reviews.get('cycle_analysis', {})

    total_lines_changed = (code_changes.get('lines_added', 0) +
                           code_changes.get('lines_removed', 0))
    average_pr_size = (total_lines_changed / prs.get('total_prs', 1)
                       if prs.get('total_prs', 0) > 0 else 0)

    pr_metrics = {
        'average_pr_size': round(average_pr_size, 2),
        'merge_rate': prs.get('merge_success_rate', 0),
        'average_merge_time': prs.get('merge_time_avg', 0),
        'review_cycles_average': cycle_analysis.get('average_cycles', 0)
    }

    review_metrics = {
        'reviews_given': code_reviews.get('total_reviews', 0),
        'average_review_length': code_reviews.get('avg_review_length', 0),
        'review_approval_rate': cycle_analysis.get('efficiency_score', 0) * 10  # Convert to percentage
    }

    collab_efficiency = {
        'response_time_avg': code_reviews.get('response_time_avg', 0),
        'iterations_per_pr': cycle_analysis.get('average_cycles', 0),
        'single_review_percentage': cycle_analysis.get('single_review_percentage', 0),
        'multiple_review_percentage': cycle_analysis.get('multiple_review_percentage', 0)
    }

    return {
        'pr_metrics': pr_metrics,
        'review_metrics': review_metrics,
        'collaboration_efficiency': collab_efficiency
    }


def _calculate_network_density(relationships: Dict[str, int], total_collaborators: int) -> float:
    if total_collaborators <= 1:
        return 0.0

    actual_connections = len(relationships)
    possible_connections = (total_collaborators * (total_collaborators - 1)) / 2

    return round((actual_connections / possible_connections) * 10, 2) if possible_connections > 0 else 0.0


def _analyze_collaboration_network(network: Dict[str, Any]) -> Dict[str, Any]:
    collaborators = network.get('collaborators', [])
    relationships = network.get('review_relationships', {})

    return {
        'network_density': _calculate_network_density(relationships, len(collaborators)),
        'collaboration_strength': {
            'strong_connections': len([v for v in relationships.values() if v >= 10]),
            'moderate_connections': len([v for v in relationships.values() if 5 <= v < 10]),
            'light_connections': len([v for v in relationships.values() if 0 < v < 5])
        },
        'key_collaborators': [
            {
                'username': user,
                'interaction_score': count,
                'relationship_strength': 'Strong' if count >= 10 else 'Moderate' if count >= 5 else 'Light'
            }
            for user, count in sorted(relationships.items(), key=lambda x: x[1], reverse=True)[:5]
        ]
    }

def _identify_mentorship_areas(metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    areas = []

    for lang, pct in metrics['languages']['primary_languages']:
        if pct >= 20:  # Expert level
            areas.append({
                'area': f'{lang} Development',
                'strength': 'Expert',
                'evidence': f'{pct:.1f}% codebase contribution'
            })

    if metrics['code_reviews']['cycle_analysis']['efficiency_score'] >= 9:
        areas.append({
            'area': 'Code Review',
            'strength': 'Strong',
            'evidence': f"{metrics['code_reviews']['total_reviews']} reviews with high efficiency"
        })

    return areas


def _calculate_repository_influence(review_activity: Dict[str, Any]) -> Dict[str, Any]:
    high_impact_repos = []
    medium_impact_repos = []
    light_impact_repos = []

    for repo, data in review_activity.items():
        impact_score = (data.get('reviews_given', 0) + data.get('reviews_received', 0)) / 2

        if impact_score >= 10:
            high_impact_repos.append(repo)
        elif impact_score >= 5:
            medium_impact_repos.append(repo)
        elif impact_score > 0:
            light_impact_repos.append(repo)

    return {
        'impact_distribution': {
            'high_impact': len(high_impact_repos),
            'medium_impact': len(medium_impact_repos),
            'light_impact': len(light_impact_repos)
        },
        'key_repositories': high_impact_repos[:5],
        'influence_score': round(
            (len(high_impact_repos) * 3 +
             len(medium_impact_repos) * 2 +
             len(light_impact_repos)) / max(len(review_activity), 1) * 10, 2
        )
    }


def _calculate_mentor_potential(metrics: Dict[str, Any]) -> Dict[str, Any]:
    reviews = metrics['code_reviews']

    return {
        'mentor_score': round(min(10, (
                (reviews['total_reviews'] / 400) * 5 +  # Review volume
                (reviews['cycle_analysis']['efficiency_score']) * 3 +  # Review quality
                (len(metrics['languages']['primary_languages']) / 5) * 2  # Language breadth
        )), 2),
        'mentorship_areas': _identify_mentorship_areas(metrics),
        'mentoring_capacity': 'High' if reviews['total_reviews'] > 300 else 'Medium' if reviews['total_reviews'] > 150 else 'Growing'
    }


def _determine_influence_level(metrics: Dict[str, Any]) -> str:
    impact_score = metrics['code_changes']['impact_score']
    review_count = metrics['code_reviews']['total_reviews']

    if impact_score >= 90 and review_count >= 300:
        return 'System-wide Influence'
    elif impact_score >= 75 and review_count >= 200:
        return 'Strong Team Influence'
    elif impact_score >= 60 and review_count >= 100:
        return 'Growing Influence'
    return 'Building Influence'


def _calculate_total_impact(metrics: Dict[str, Any]) -> float:
    code_impact = metrics['code_changes']['impact_score']
    pr_impact = (metrics['pull_requests']['merged_prs'] /
                max(metrics['pull_requests']['total_prs'], 1)) * 100
    review_impact = metrics['code_reviews'].get('total_reviews', 0) * 0.5

    return code_impact * 0.4 + pr_impact * 0.3 + review_impact * 0.3


def _analyze_contribution_impact(metrics: Dict[str, Any]) -> Dict[str, Any]:
    code_changes = metrics['code_changes']
    reviews = metrics['code_reviews']

    return {
        'impact_metrics': {
            'code_changes': code_changes['total_commits'],
            'lines_changed': code_changes['lines_added'] + code_changes['lines_removed'],
            'reviews_given': reviews['total_reviews']
        },
        'impact_score': round(
            (code_changes['impact_score'] * 0.6 +
             reviews['cycle_analysis']['efficiency_score'] * 0.4), 2
        ),
        'influence_level': _determine_influence_level(metrics)
    }


def _analyze_language_category(distribution: Dict[str, float], category_languages: List[str]) -> Dict[
    str, Any]:
    category_usage = {
        lang: distribution.get(lang, 0)
        for lang in category_languages
        if distribution.get(lang, 0) > 0
    }

    if not category_usage:
        return {
            'active_languages': [],
            'primary_language': None,
            'total_usage': 0
        }

    total_usage = sum(category_usage.values())
    primary_lang = max(category_usage.items(), key=lambda x: x[1])[0] if category_usage else None

    return {
        'active_languages': list(category_usage.keys()),
        'primary_language': primary_lang,
        'total_usage': round(total_usage, 2)
    }

def _get_primary_languages(languages: Dict[str, Any]) -> List[tuple[str, float]]:
    distribution = languages.get('language_distribution', {})
    return sorted(distribution.items(), key=lambda x: x[1], reverse=True)[:5]

def _calculate_language_diversity(languages: Dict[str, Any]) -> float:
    distribution = languages.get('language_distribution', {})
    if not distribution:
        return 0.0

    total = sum(distribution.values())
    if total == 0:
        return 0.0

    # Calculate normalized entropy as a diversity measure
    proportions = [v/total for v in distribution.values()]
    entropy = -sum(p * np.log(p) if p > 0 else 0 for p in proportions)
    max_entropy = np.log(len(distribution)) if len(distribution) > 0 else 1

    return (entropy / max_entropy if max_entropy > 0 else 0.0) * 10  # Scale to 0-10

def _determine_expertise_level(usage_percentage: float) -> str:
    if usage_percentage >= 20:
        return "Expert"
    elif usage_percentage >= 10:
        return "Advanced"
    elif usage_percentage >= 5:
        return "Intermediate"
    elif usage_percentage >= 1:
        return "Beginner"
    return "Novice"

def _calculate_core_team_interaction(collaboration: Dict[str, Any]) -> Dict[str, float]:
    network = collaboration.get('collaboration_network', {})
    collaborators = network.get('collaborators', [])
    review_relationships = network.get('review_relationships', {})

    if not collaborators or not review_relationships:
        return {
            'interaction_score': 0.0,
            'team_cohesion': 0.0,
            'core_contributor_count': 0
        }

    # Identify core contributors (those with significant interactions)
    interaction_threshold = 5  # Minimum interactions to be considered 'core'
    core_contributors = [
        collab for collab in collaborators
        if review_relationships.get(collab, 0) >= interaction_threshold
    ]

    # Calculate interaction score
    total_interactions = sum(review_relationships.values())
    core_interactions = sum(
        review_relationships.get(collab, 0)
        for collab in core_contributors
    )

    interaction_score = (core_interactions / total_interactions * 10) if total_interactions > 0 else 0

    # Calculate team cohesion (how evenly distributed the interactions are)
    if core_contributors:
        interaction_values = [review_relationships.get(collab, 0) for collab in core_contributors]
        mean_interactions = sum(interaction_values) / len(interaction_values)
        variance = sum((x - mean_interactions) ** 2 for x in interaction_values) / len(interaction_values)
        team_cohesion = 10 / (1 + variance)  # Normalize to 0-10 scale, higher variance = lower cohesion
    else:
        team_cohesion = 0

    return {
        'interaction_score': round(interaction_score, 2),
        'team_cohesion': round(team_cohesion, 2),
        'core_contributor_count': len(core_contributors)
    }

def _calculate_expertise_distribution(language_distribution: Dict[str, float]) -> Dict[str, int]:
    expertise_counts = {
        'Expert': 0,
        'Advanced': 0,
        'Intermediate': 0,
        'Beginner': 0,
        'Novice': 0
    }

    for percentage in language_distribution.values():
        level = _determine_expertise_level(percentage)
        expertise_counts[level] += 1

    return expertise_counts


def _calculate_language_expertise(languages: Dict[str, Any]) -> Dict[str, str]:
    distribution = languages.get('language_distribution', {})
    return {
        lang: _determine_expertise_level(percentage)
        for lang, percentage in distribution.items()
    }


def _calculate_cross_team_collaboration(collaboration: Dict[str, Any]) -> Dict[str, float]:
    network = collaboration.get('collaboration_network', {})
    cross_repo = collaboration.get('cross_repo_impact', {})

    repos = set(cross_repo.get('repositories', []))
    collaborators = set(network.get('collaborators', []))

    if not repos or not collaborators:
        return {
            'collaboration_score': 0.0,
            'repository_spread': 0.0,
            'unique_collaborators': 0
        }

    repo_count = len(repos)
    collab_count = len(collaborators)

    # Calculate collaboration score (0-10 scale)
    # Factor in both number of repos and collaborators
    collaboration_score = min(10.0, (repo_count * collab_count) ** 0.5)

    # Calculate repository spread (average collaborators per repository)
    repository_spread = collab_count / repo_count if repo_count > 0 else 0

    return {
        'collaboration_score': round(collaboration_score, 2),
        'repository_spread': round(repository_spread, 2),
        'unique_collaborators': collab_count
    }


def _calculate_review_distribution(collaboration: Dict[str, Any]) -> Dict[str, Any]:
    # REF:
    #  - https://en.wikipedia.org/wiki/Lorenz_curve
    #  - https://en.wikipedia.org/wiki/Income_inequality_metrics
    network = collaboration.get('collaboration_network', {})
    review_relationships = network.get('review_relationships', {})

    if not review_relationships:
        return {
            'distribution_score': 0.0,
            'review_concentration': 0.0,
            'reviewer_stats': {
                'total_reviewers': 0,
                'active_reviewers': 0,
                'most_active_count': 0
            }
        }

    review_counts = dict(review_relationships)
    total_reviews = sum(review_counts.values())
    total_reviewers = len(review_counts)

    if total_reviews == 0 or total_reviewers == 0:
        return {
            'distribution_score': 0.0,
            'review_concentration': 0.0,
            'reviewer_stats': {
                'total_reviewers': 0,
                'active_reviewers': 0,
                'most_active_count': 0
            }
        }

    # Calculate distribution score using Lorenz curve area comparison
    sorted_counts = sorted(review_counts.values())
    cumsum = 0
    lorenz_sum = 0
    for i, count in enumerate(sorted_counts):
        cumsum += count
        lorenz_sum += (i + 1) * count

    # Calculate area between Lorenz curve and line of perfect equality
    perfect_equality_area = total_reviews * total_reviewers / 2
    actual_area = cumsum * total_reviewers - lorenz_sum

    # Convert to a 0-10 scale where higher values mean more equal distribution
    distribution_score = (actual_area / perfect_equality_area) * 10 if perfect_equality_area > 0 else 0

    # Calculate review concentration (what % of reviews are done by top 20% of reviewers aka quintile analysis)
    top_reviewer_count = max(1, total_reviewers // 5)
    top_reviewer_reviews = sum(sorted(review_counts.values(), reverse=True)[:top_reviewer_count])
    review_concentration = (top_reviewer_reviews / total_reviews * 100) if total_reviews > 0 else 0

    # Count active reviewers (more than 1 review)
    active_reviewers = sum(1 for count in review_counts.values() if count > 1)
    most_active_count = max(review_counts.values()) if review_counts else 0

    return {
        'distribution_score': round(distribution_score, 2),  # 0-10 scale, higher is more evenly distributed
        'review_concentration': round(review_concentration, 2),  # Percentage of reviews by top quintile
        'reviewer_stats': {
            'total_reviewers': total_reviewers,
            'active_reviewers': active_reviewers,
            'most_active_count': most_active_count
        }
    }

def _analyze_language_patterns(metrics: Dict[str, Any]) -> Dict[str, Any]:
    languages = metrics.get('languages', {})
    distribution = languages.get('language_distribution', {})

    expertise_levels = _calculate_language_expertise(languages)
    expertise_distribution = _calculate_expertise_distribution(distribution)

    language_summary = {
        'total_languages': len(distribution),
        'major_languages': len([lang for lang, pct in distribution.items() if pct >= 5]),
        'top_language': _get_primary_languages(languages)[0] if distribution else ('None', 0),
        'expertise_distribution': expertise_distribution,
        'expertise_by_category': {
            'backend': _analyze_language_category(distribution, ['Go', 'Python']),
            'frontend': _analyze_language_category(distribution, ['JavaScript', 'TypeScript', 'HTML', 'CSS']),
            'infrastructure': _analyze_language_category(distribution, ['Dockerfile', 'Shell', 'Makefile']),
            'data': _analyze_language_category(distribution, ['SQL', 'PLpgSQL'])
        }
    }

    return {
        'primary_languages': _get_primary_languages(languages)[:5],
        'expertise_levels': expertise_levels,
        'language_diversity': _calculate_language_diversity(languages),
        'language_summary': language_summary
    }

def _analyze_code_velocity_trend(metrics: Dict[str, Any]) -> Dict[str, Any]:
    code_changes = metrics['code_changes']
    prs = metrics['pull_requests']
    reviews = metrics['code_reviews']

    total_commits = code_changes['total_commits']
    daily_commit_avg = total_commits / 365
    weekly_commit_avg = daily_commit_avg * 7

    total_prs = prs['total_prs']
    daily_pr_avg = total_prs / 365

    lines_per_day = (code_changes['lines_added'] + code_changes['lines_removed']) / 365
    files_per_day = code_changes['files_changed'] / 365

    return {
        'commit_velocity': {
            'current': total_commits,
            'daily_average': round(daily_commit_avg, 2),
            'weekly_average': round(weekly_commit_avg, 2),
            'distribution': code_changes['commit_activity']['day_distribution'],
            'intensity': {
                'lines_per_commit': round(
                    (code_changes['lines_added'] + code_changes['lines_removed']) / total_commits, 2),
                'files_per_commit': round(code_changes['files_changed'] / total_commits, 2)
            }
        },
        'pr_velocity': {
            'total_prs': total_prs,
            'daily_average': round(daily_pr_avg, 2),
            'merge_success_rate': round(prs['merge_success_rate'], 2),
            'average_merge_time_hours': round(prs['merge_time_avg'], 2),
            'efficiency': {
                'review_cycles_avg': round(reviews['cycle_analysis']['average_cycles'], 2),
                'first_response_time_avg': round(reviews['response_time_avg'] * 24, 2),  # Convert to hours...
                'single_review_ratio': round(reviews['cycle_analysis']['single_review_percentage'], 2)
            }
        },
        'work_intensity': {
            'daily_output': {
                'lines_changed': round(lines_per_day, 2),
                'files_modified': round(files_per_day, 2),
                'reviews_given': round(reviews['total_reviews'] / 365, 2)
            },
            'peak_activity': {
                'day': max(code_changes['commit_activity']['day_distribution'].items(), key=lambda x: x[1])[0],
                'hour': max(code_changes['commit_activity']['hour_distribution'].items(), key=lambda x: x[1])[0],
                'commits_at_peak': max(code_changes['commit_activity']['day_distribution'].values())
            },
            'repository_impact': {
                'total_repos': len(code_changes['repositories_processed']),
                'avg_commits_per_repo': round(total_commits / len(code_changes['repositories_processed']), 2)
            }
        }
    }


def _determine_language_contribution_type(language: str, metrics: Dict[str, Any]) -> str:
    repos = metrics['code_changes']['repositories_processed']
    lang_repos = [
        repo for repo in repos
        if repo['name'] in metrics['collaboration']['cross_repo_impact']['review_activity']
           and metrics['collaboration']['cross_repo_impact']['review_activity'][repo['name']][
               'primary_language'] == language
    ]

    if len(lang_repos) > 5:
        return "Core Contributor"
    elif len(lang_repos) > 2:
        return "Regular Contributor"
    return "Occasional Contributor"


def _calculate_language_impact_level(language: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
    review_activity = metrics['collaboration']['cross_repo_impact']['review_activity']

    lang_repos = [
        repo for repo in review_activity.values()
        if repo['primary_language'] == language
    ]

    total_reviews = sum(repo['reviews_given'] + repo['reviews_received'] for repo in lang_repos)

    return {
        'repositories': len(lang_repos),
        'total_reviews': total_reviews,
        'impact_rating': 'High' if total_reviews > 50 else 'Medium' if total_reviews > 20 else 'Growing'
    }


def _calculate_language_category_usage(dist: Dict[str, float], category_langs: List[str]) -> Dict[str, Any]:
    category_usage = sum(dist.get(lang, 0) for lang in category_langs)
    active_langs = [lang for lang in category_langs if dist.get(lang, 0) >= 1.0]

    return {
        'total_usage': round(category_usage, 2),
        'active_languages': active_langs,
        'primary_language': max(((lang, dist.get(lang, 0))
                                 for lang in active_langs),
                                key=lambda x: x[1])[0] if active_langs else None
    }


def _calculate_interaction_strength(network: Dict[str, Any]) -> Dict[str, Any]:
    relationships = network['review_relationships']
    if not relationships:
        return {'score': 0, 'level': 'No Interaction'}

    total_interactions = sum(relationships.values())
    unique_interactions = len(relationships)
    avg_interaction_strength = total_interactions / unique_interactions if unique_interactions > 0 else 0

    return {
        'score': round(avg_interaction_strength, 2),
        'level': (
            'Very Strong' if avg_interaction_strength > 10
            else 'Strong' if avg_interaction_strength > 5
            else 'Moderate' if avg_interaction_strength > 2
            else 'Growing'
        )
    }


def _analyze_review_relationships(relationships: Dict[str, int]) -> Dict[str, Any]:
    if not relationships:
        return {'strength': 'No Relationships', 'distribution': {}}

    sorted_relationships = sorted(relationships.items(), key=lambda x: x[1], reverse=True)

    return {
        'strongest_relationships': sorted_relationships[:3],
        'relationship_distribution': {
            'strong': len([v for v in relationships.values() if v > 10]),
            'moderate': len([v for v in relationships.values() if 5 < v <= 10]),
            'light': len([v for v in relationships.values() if v <= 5])
        }
    }


def _calculate_repo_impact_distribution(review_activity: Dict[str, Any]) -> Dict[str, Any]:
    impact_levels = {
        'high_impact': 0,
        'medium_impact': 0,
        'light_impact': 0
    }

    for repo in review_activity.values():
        total_reviews = repo['reviews_given'] + repo['reviews_received']
        if total_reviews > 20:
            impact_levels['high_impact'] += 1
        elif total_reviews > 10:
            impact_levels['medium_impact'] += 1
        elif total_reviews > 0:
            impact_levels['light_impact'] += 1

    return impact_levels


def _determine_collaboration_level(
        review_engagement: Dict[str, Any],
        team_interaction: Dict[str, Any],
        repo_impact: Dict[str, Any]
) -> Dict[str, Any]:
    strengths = []
    if review_engagement['total_reviews'] > 400:
        strengths.append('High Review Volume')
    if review_engagement['review_efficiency']['score'] > 9:
        strengths.append('Efficient Review Process')
    if team_interaction['collaboration_breadth']['total_collaborators'] > 20:
        strengths.append('Broad Team Impact')
    if repo_impact['total_repositories'] > 20:
        strengths.append('Wide Repository Reach')

    return {
        'level': (
            'Exceptional' if len(strengths) >= 3
            else 'Very High' if len(strengths) >= 2
            else 'High' if len(strengths) >= 1
            else 'Growing'
        ),
        'strengths': strengths
    }


def _calculate_overall_impact_score(metrics: Dict[str, Any]) -> float:
    # Normalizing all metrics...

    # Code Volume Impact (30%)
    code_volume_score = min(100, (
            (metrics['code_changes']['total_commits'] / 1000) * 60 +
            (len(metrics['code_changes']['repositories_processed']) / 20) * 40
    ))

    # Code Quality Impact (25%)
    code_quality_score = (
            metrics['pull_requests']['merge_success_rate'] * 0.6 +
            metrics['code_reviews']['cycle_analysis']['efficiency_score'] * 10 * 0.4
    )

    # Review Impact (25%)
    review_impact_score = min(100, (
            (metrics['code_reviews']['total_reviews'] / 300) * 60 +
            metrics['code_reviews']['cycle_analysis']['single_review_percentage'] * 0.4
    ))

    # Collaboration Impact (20%)
    collab_impact_score = min(100, (
            (metrics['collaboration']['cross_repo_impact']['impact_score'] * 0.7) +  # Cross-repo impact
            (metrics['code_reviews']['cross_repo_reviews']['review_spread'] * 100 * 0.3)  # Review distribution
    ))

    # Calculate a weighted average
    overall_score = (
            code_volume_score * 0.30 +
            code_quality_score * 0.25 +
            review_impact_score * 0.25 +
            collab_impact_score * 0.20
    )

    return round(overall_score, 2)


def _identify_primary_impact_areas(metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    impact_areas = []

    # Code Volume Impact
    commits = metrics['code_changes']['total_commits']
    if commits > 1500:
        impact_areas.append({
            'area': 'Code Contributions',
            'strength': 'Exceptional',
            'evidence': f"{commits} commits across {len(metrics['code_changes']['repositories_processed'])} repositories",
            'percentile': '95th+'
        })
    elif commits > 750:
        impact_areas.append({
            'area': 'Code Contributions',
            'strength': 'Very High',
            'evidence': f"Significant volume with {commits} commits",
            'percentile': '90th'
        })
    elif commits > 300:
        impact_areas.append({
            'area': 'Code Contributions',
            'strength': 'High',
            'evidence': f"Strong contribution with {commits} commits",
            'percentile': '75th'
        })
    elif commits > 100:
        impact_areas.append({
            'area': 'Code Contributions',
            'strength': 'Growing',
            'evidence': f"Regular contributor with {commits} commits",
            'percentile': '50th'
        })

    # Review Impact
    reviews = metrics['code_reviews']['total_reviews']
    efficiency = metrics['code_reviews']['cycle_analysis']['efficiency_score']
    if reviews > 400 and efficiency > 9.5:
        impact_areas.append({
            'area': 'Code Reviews',
            'strength': 'Exceptional',
            'evidence': f"{reviews} reviews with {efficiency:.2f}/10 efficiency",
            'percentile': '95th+'
        })
    elif reviews > 200 or efficiency > 9.0:
        impact_areas.append({
            'area': 'Code Reviews',
            'strength': 'Very High',
            'evidence': f"Major reviewer with {reviews} reviews",
            'percentile': '90th'
        })
    elif reviews > 100 or efficiency > 8.5:
        impact_areas.append({
            'area': 'Code Reviews',
            'strength': 'High',
            'evidence': f"Active reviewer with {reviews} reviews",
            'percentile': '75th'
        })
    elif reviews > 50:
        impact_areas.append({
            'area': 'Code Reviews',
            'strength': 'Growing',
            'evidence': f"Regular reviewer with {reviews} reviews",
            'percentile': '50th'
        })

    # Cross-Repo Impact
    repos = len(metrics['collaboration']['cross_repo_impact']['repositories'])
    if repos > 20:
        impact_areas.append({
            'area': 'Cross-Repository Impact',
            'strength': 'Exceptional',
            'evidence': f"System-wide impact across {repos} repositories",
            'percentile': '95th+'
        })
    elif repos > 15:
        impact_areas.append({
            'area': 'Cross-Repository Impact',
            'strength': 'Very High',
            'evidence': f"Broad impact across {repos} repositories",
            'percentile': '90th'
        })
    elif repos > 10:
        impact_areas.append({
            'area': 'Cross-Repository Impact',
            'strength': 'High',
            'evidence': f"Strong presence in {repos} repositories",
            'percentile': '75th'
        })
    elif repos > 5:
        impact_areas.append({
            'area': 'Cross-Repository Impact',
            'strength': 'Growing',
            'evidence': f"Growing impact across {repos} repositories",
            'percentile': '50th'
        })

    # PR Quality Impact
    pr_success = metrics['pull_requests']['merge_success_rate']
    if pr_success > 95:
        impact_areas.append({
            'area': 'PR Quality',
            'strength': 'Exceptional',
            'evidence': f"{pr_success:.1f}% PR success rate",
            'percentile': '95th+'
        })
    elif pr_success > 90:
        impact_areas.append({
            'area': 'PR Quality',
            'strength': 'Very High',
            'evidence': f"{pr_success:.1f}% PR success rate",
            'percentile': '90th'
        })
    elif pr_success > 85:
        impact_areas.append({
            'area': 'PR Quality',
            'strength': 'High',
            'evidence': f"{pr_success:.1f}% PR success rate",
            'percentile': '75th'
        })
    elif pr_success > 75:
        impact_areas.append({
            'area': 'PR Quality',
            'strength': 'Growing',
            'evidence': f"{pr_success:.1f}% PR success rate",
            'percentile': '50th'
        })

    return impact_areas


def _generate_achievements(metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    achievements = []

    code_changes = metrics.get('code_changes', {})
    reviews = metrics.get('code_reviews', {})
    prs = metrics.get('pull_requests', {})
    languages = metrics.get('languages', {})
    collaboration = metrics.get('collaboration', {})

    if code_changes['total_commits'] >= 1500:
        achievements.append({
            'title': 'Code Machine',
            'description': f"Made over {code_changes['total_commits']} commits in a year",
            'level': 'platinum',
            'icon': '🏆'
        })
    elif code_changes['total_commits'] >= 500:
        achievements.append({
            'title': 'Code Machine',
            'description': 'Made over 500 commits in a year',
            'level': 'gold',
            'icon': '🏆'
        })

    if reviews['total_reviews'] >= 400:
        achievements.append({
            'title': 'Master Reviewer',
            'description': f"Provided {reviews['total_reviews']} high-quality code reviews",
            'level': 'platinum',
            'icon': '👀'
        })
    elif reviews['total_reviews'] >= 100:
        achievements.append({
            'title': 'Eagle Eye',
            'description': 'Provided over 100 code reviews',
            'level': 'gold',
            'icon': '👀'
        })

    lang_dist = languages['language_distribution']
    active_languages = len([lang for lang, pct in lang_dist.items() if pct >= 5])
    if active_languages >= 5:
        achievements.append({
            'title': 'Polyglot Developer',
            'description': 'Actively used more than 5 programming languages',
            'level': 'gold',
            'icon': '🌎'
        })

    pr_success_rate = prs.get('merge_success_rate', 0)
    if pr_success_rate >= 95:
        achievements.append({
            'title': 'Quality Champion',
            'description': f"Maintained {pr_success_rate:.1f}% PR success rate",
            'level': 'gold',
            'icon': '🎯'
        })

    if code_changes['impact_score'] >= 90:
        achievements.append({
            'title': 'High Impact Developer',
            'description': 'Demonstrated exceptional code impact',
            'level': 'gold',
            'icon': '💫'
        })

    backend_usage = sum(
        pct for lang, pct in lang_dist.items()
        if lang in ['Go', 'Python']
    )
    if backend_usage >= 80:
        achievements.append({
            'title': 'Backend Specialist',
            'description': 'Demonstrated strong backend development expertise',
            'level': 'gold',
            'icon': '⚙️'
        })

    cross_team_impact = collaboration['cross_repo_impact']
    if cross_team_impact['impact_score'] >= 90:
        achievements.append({
            'title': 'Team Amplifier',
            'description': 'Demonstrated exceptional cross-team impact',
            'level': 'gold',
            'icon': '🤝'
        })

    commits_per_day = code_changes['total_commits'] / 365
    if commits_per_day >= 4:
        achievements.append({
            'title': 'Velocity Master',
            'description': f"Averaged {commits_per_day:.1f} commits per day",
            'level': 'gold',
            'icon': '🚀'
        })

    commit_activity = code_changes.get('commit_activity', {})
    day_distribution = commit_activity.get('day_distribution', {})
    max_day_commits = max(day_distribution.values())
    if max_day_commits >= 300:
        achievements.append({
            'title': 'Peak Performer',
            'description': f"Achieved {max_day_commits} commits in a single day",
            'level': 'gold',
            'icon': '🏔️'
        })

    collaboration_network = collaboration.get('collaboration_network', {})
    collaborators = len(collaboration_network.get('collaborators', []))
    if collaborators >= 20:
        achievements.append({
            'title': 'Network Builder',
            'description': f'Collaborated with {collaborators} different developers',
            'level': 'gold',
            'icon': '🌐'
        })

    top_lang_usage = languages['primary_languages'][0][1]  # Usage % of top language
    if top_lang_usage >= 60:
        achievements.append({
            'title': 'Core Technology Expert',
            'description': f'Mastered primary technology with {top_lang_usage:.1f}% usage',
            'level': 'platinum',
            'icon': '🎓'
        })

    if reviews['cycle_analysis']['single_review_percentage'] >= 95:
        achievements.append({
            'title': 'One-Shot Wonder',
            'description': 'Over 95% PRs approved in single review cycle',
            'level': 'platinum',
            'icon': '🎯'
        })

    high_impact_repos = sum(1 for repo in metrics['code_changes']['repositories_processed']
                            if repo['commits'] >= 100)

    if reviews['cross_repo_reviews']['unique_repositories'] >= 20:
        achievements.append({
            'title': 'Cross-Repository Guardian',
            'description': f'Reviewed code across {reviews["cross_repo_reviews"]["unique_repositories"]} repositories',
            'level': 'platinum',
            'icon': '🛡️'
        })

    avg_change_size = (code_changes['lines_added'] + code_changes['lines_removed']) / code_changes['total_commits']
    if avg_change_size <= 200 and code_changes['total_commits'] >= 1000:
        achievements.append({
            'title': 'Precision Engineer',
            'description': 'Maintained focused commit sizes with high volume',
            'level': 'gold',
            'icon': '🎯'
        })

    weekend_commits = day_distribution.get('Saturday', 0) + day_distribution.get('Sunday', 0)
    if weekend_commits >= 100:
        achievements.append({
            'title': 'Weekend Warrior',
            'description': f'Made {weekend_commits} contributions on weekends',
            'level': 'gold',
            'icon': '⚔️'
        })

    evening_commits = sum(
        commits for hour, commits in code_changes['commit_activity']['hour_distribution'].items()
        if hour >= 18 or hour <= 5
    )
    if evening_commits >= 500:
        achievements.append({
            'title': 'Night Owl',
            'description': f'Made {evening_commits} contributions during evening hours',
            'level': 'gold',
            'icon': '🦉'
        })

    balanced_languages = len([pct for _, pct in languages['primary_languages'] if pct >= 10])
    if balanced_languages >= 3:
        achievements.append({
            'title': 'Technology Balanced',
            'description': f'Maintained significant usage across {balanced_languages} languages',
            'level': 'gold',
            'icon': '⚖️'
        })

    if reviews['review_comments'] >= 100:
        achievements.append({
            'title': 'Thorough Reviewer',
            'description': f'Provided {reviews["review_comments"]} detailed review comments',
            'level': 'gold',
            'icon': '📝'
        })

    consistency_score = code_changes['commit_activity']['consistency_score']
    if consistency_score >= 0.3:
        achievements.append({
            'title': 'Consistent Contributor',
            'description': 'Maintained steady contribution pattern',
            'level': 'gold',
            'icon': '📊'
        })

    infra_usage = sum(
        pct for lang, pct in languages['language_distribution'].items()
        if lang in ['Dockerfile', 'Shell', 'Makefile']
    )
    if infra_usage >= 5:
        achievements.append({
            'title': 'Infrastructure Specialist',
            'description': 'Demonstrated infrastructure expertise',
            'level': 'gold',
            'icon': '🏗️'
        })

    if reviews['total_reviews'] >= 300 and reviews['review_comments'] >= 100:
        achievements.append({
            'title': 'Knowledge Amplifier',
            'description': 'Exceptional contribution to team learning through detailed reviews',
            'level': 'platinum',
            'icon': '📚'
        })

    friday_commits = day_distribution.get('Friday', 0)
    if friday_commits >= 250:
        achievements.append({
            'title': 'Sprint Finisher',
            'description': f'Delivered {friday_commits} Friday contributions',
            'level': 'gold',
            'icon': '🏃'
        })

    backend_usage = sum(pct for lang, pct in languages['language_distribution'].items()
                        if lang in ['Go', 'Python'])
    frontend_usage = sum(pct for lang, pct in languages['language_distribution'].items()
                         if lang in ['JavaScript', 'TypeScript', 'HTML', 'CSS'])
    if backend_usage >= 20 and frontend_usage >= 5:
        achievements.append({
            'title': 'Full Stack Developer',
            'description': 'Demonstrated expertise across frontend and backend',
            'level': 'gold',
            'icon': '🏗️'
        })

    if (reviews['response_time_avg'] <= 0.05 and  # Less than 1.2 hours
            reviews['total_reviews'] >= 200):
        achievements.append({
            'title': 'Lightning Reviewer',
            'description': 'Consistently provided rapid, high-quality reviews',
            'level': 'platinum',
            'icon': '⚡'
        })

    test_commits = sum(1 for repo in metrics['code_changes']['repositories_processed']
                       if any('test' in commit.lower()
                              for commit in repo.get('commit_messages', [])))
    if test_commits >= 100:
        achievements.append({
            'title': 'Testing Champion',
            'description': 'Strong focus on code quality and testing',
            'level': 'gold',
            'icon': '🎯'
        })

    if code_changes['total_commits'] >= 1000 and pr_success_rate >= 90:
        achievements.append({
            'title': 'Feature Velocity Master',
            'description': 'High commit volume with excellent success rate',
            'level': 'platinum',
            'icon': '🚀'
        })

    if (reviews['total_reviews'] >= 200 and
            reviews['cycle_analysis']['efficiency_score'] >= 9.0):
        achievements.append({
            'title': 'Technical Mentor',
            'description': 'Demonstrated strong mentorship potential',
            'level': 'gold',
            'icon': '👨‍🏫'
        })

    diverse_contributions = {
        'backend': backend_usage >= 20,
        'frontend': frontend_usage >= 5,
        'infra': infra_usage >= 3,
        'reviews': reviews['total_reviews'] >= 100
    }
    if sum(diverse_contributions.values()) >= 3:
        achievements.append({
            'title': 'Cross-Functional Expert',
            'description': 'Demonstrated expertise across multiple domains',
            'level': 'platinum',
            'icon': '🔄'
        })

    repos = metrics['collaboration']['cross_repo_impact']['repositories']

    if high_impact_repos >= 3:
        achievements.append({
            'title': 'Repository Leader',
            'description': f'Major contributor in {high_impact_repos} repositories',
            'level': 'platinum',
            'icon': '👑'
        })

    if len(repos) >= 5 and high_impact_repos >= 3:
        achievements.append({
            'title': 'Repository Pioneer',
            'description': 'Drove adoption and growth across repositories',
            'level': 'platinum',
            'icon': '🌱'
        })

    if (pr_success_rate >= 95 and
            reviews['cycle_analysis']['single_review_percentage'] >= 90):
        achievements.append({
            'title': 'Quality Guardian',
            'description': 'Maintained exceptional code quality standards',
            'level': 'platinum',
            'icon': '🛡️'
        })

    if (len(repos) >= 20 and
            code_changes['impact_score'] >= 90 and
            high_impact_repos >= 5):
        achievements.append({
            'title': 'System Architect',
            'description': 'Demonstrated system-wide architectural impact',
            'level': 'platinum',
            'icon': '🏛️'
        })

    if (reviews['total_reviews'] >= 400 and
            reviews['cycle_analysis']['efficiency_score'] >= 9.5):
        achievements.append({
            'title': 'Review Culture Champion',
            'description': 'Exceptional contribution to review culture',
            'level': 'platinum',
            'icon': '⚡'
        })

    max_commits_in_hour = max(
        commits for hour, commits in
        code_changes['commit_activity']['hour_distribution'].items()
    )
    if max_commits_in_hour >= 50:
        achievements.append({
            'title': 'Breakthrough Achiever',
            'description': f'Achieved {max_commits_in_hour} commits in a single hour',
            'level': 'gold',
            'icon': '💥'
        })

    high_impact_score = metrics['collaboration']['cross_repo_impact']['impact_score']
    if (high_impact_score >= 90 and
            reviews['total_reviews'] >= 300 and
            code_changes['total_commits'] >= 1000):
        achievements.append({
            'title': 'Technical Leader',
            'description': 'Demonstrated outstanding technical leadership',
            'level': 'platinum',
            'icon': '👑'
        })

    hour_dist = code_changes['commit_activity']['hour_distribution']

    morning_sprint_commits = sum(
        commits for hour, commits in hour_dist.items()
        if 8 <= hour <= 11
    )
    if morning_sprint_commits >= 400:
        achievements.append({
            'title': 'Morning Sprint Champion',
            'description': f'Made {morning_sprint_commits} contributions during morning hours',
            'level': 'gold',
            'icon': '☀️'
        })

    afternoon_commits = sum(
        commits for hour, commits in hour_dist.items()
        if 14 <= hour <= 16
    )
    if afternoon_commits >= 300:
        achievements.append({
            'title': 'Afternoon Push Master',
            'description': f'Made {afternoon_commits} contributions during afternoon hours',
            'level': 'gold',
            'icon': '🌤️'
        })

    return achievements


def _analyze_collaboration_patterns(metrics: Dict[str, Any]) -> Dict[str, Any]:
    collaboration = metrics['collaboration']
    network = collaboration.get('collaboration_network', {})
    review_activity = collaboration.get('cross_repo_impact', {}).get('review_activity', {})

    review_relationships = network.get('review_relationships', {})
    favorite_reviewers = sorted(
        [
            {'reviewer': reviewer, 'reviews': count}
            for reviewer, count in review_relationships.items()
        ],
        key=lambda x: x['reviews'],
        reverse=True
    )[:3]

    team_metrics = {
        'total_collaborators': len(network.get('collaborators', [])),
        'core_team_interaction': _calculate_core_team_interaction(collaboration),
        'cross_team_collaboration': _calculate_cross_team_collaboration(collaboration),
        'collaboration_network': _analyze_collaboration_network(network)
    }

    review_relationships = {
        'favorite_reviewers': favorite_reviewers,
        'review_distribution': _calculate_review_distribution(collaboration),
        'mentor_potential': _calculate_mentor_potential(metrics)
    }

    project_impact = {
        'projects_contributed': len(collaboration.get('cross_repo_impact', {}).get('repositories', [])),
        'impact_score': collaboration.get('cross_repo_impact', {}).get('impact_score', 0),
        'repository_influence': _calculate_repository_influence(review_activity),
        'contribution_impact': _analyze_contribution_impact(metrics)
    }

    return {
        'team_metrics': team_metrics,
        'review_relationships': review_relationships,
        'project_impact': project_impact
    }


def _analyze_language_growth(metrics: Dict[str, Any]) -> Dict[str, Any]:
    languages = metrics['languages']
    language_dist = languages['language_distribution']
    primary_languages = languages['primary_languages']

    # Calculate primary language metrics
    primary_language_metrics = {}
    for lang, percentage in primary_languages:
        if percentage >= 1.0:  # Only include languages with meaningful usage
            primary_language_metrics[lang] = {
                'current_usage': round(percentage, 2),
                'expertise_level': _determine_expertise_level(percentage),
                'contribution_type': _determine_language_contribution_type(lang, metrics),
                'impact_level': _calculate_language_impact_level(lang, metrics)
            }

    # Calculate language diversity metrics
    diversity_metrics = {
        'total_languages': len(language_dist),
        'active_languages': len([lang for lang, pct in language_dist.items() if pct >= 1.0]),
        'primary_languages': len([lang for lang, pct in language_dist.items() if pct >= 5.0]),
        'diversity_score': round(_calculate_language_diversity(languages), 2)
    }

    return {
        'primary_languages': primary_language_metrics,
        'diversity_metrics': diversity_metrics,
        'expertise_distribution': _calculate_expertise_distribution(language_dist),
        'language_focus': {
            'backend': _calculate_language_category_usage(language_dist, ['Go', 'Python']),
            'frontend': _calculate_language_category_usage(language_dist,
                                                                ['JavaScript', 'TypeScript', 'HTML', 'CSS']),
            'infrastructure': _calculate_language_category_usage(language_dist,
                                                                      ['Dockerfile', 'Shell', 'Makefile']),
            'data': _calculate_language_category_usage(language_dist, ['SQL', 'PLpgSQL'])
        }
    }


def _analyze_collaboration_growth(metrics: Dict[str, Any]) -> Dict[str, Any]:
    reviews = metrics['code_reviews']
    collaboration = metrics['collaboration']

    # Calculate review engagement metrics
    review_engagement = {
        'total_reviews': reviews['total_reviews'],
        'review_efficiency': {
            'score': reviews['cycle_analysis']['efficiency_score'],
            'single_review_success': reviews['cycle_analysis']['single_review_percentage'],
            'average_cycles': reviews['cycle_analysis']['average_cycles']
        },
        'response_patterns': {
            'average_response_time': round(reviews['response_time_avg'] * 24, 2),  # in hours
            'review_length': round(reviews['avg_review_length'], 2)
        }
    }

    # Calculate team interaction metrics
    network = collaboration['collaboration_network']
    team_interaction = {
        'collaboration_breadth': {
            'total_collaborators': len(network['collaborators']),
            'active_relationships': len(network['review_relationships']),
            'interaction_strength': _calculate_interaction_strength(network)
        },
        'review_relationships': {
            'given': sum(network['review_relationships'].values()),
            'relationships': _analyze_review_relationships(network['review_relationships'])
        }
    }

    # Calculate repository impact
    repo_impact = {
        'total_repositories': len(collaboration['cross_repo_impact']['repositories']),
        'active_repositories': len([
            repo for repo in collaboration['cross_repo_impact']['review_activity'].values()
            if repo['reviews_given'] > 0 or repo['reviews_received'] > 0
        ]),
        'impact_distribution': _calculate_repo_impact_distribution(
            collaboration['cross_repo_impact']['review_activity']
        )
    }

    return {
        'review_engagement': review_engagement,
        'team_interaction': team_interaction,
        'repository_impact': repo_impact,
        'collaboration_level': _determine_collaboration_level(
            review_engagement, team_interaction, repo_impact
        )
    }


def _analyze_impact_growth(metrics: Dict[str, Any]) -> Dict[str, Any]:
    code_changes = metrics['code_changes']
    reviews = metrics['code_reviews']
    collaboration = metrics['collaboration']

    code_impact = {
        'commit_impact': {
            'total_commits': code_changes['total_commits'],
            'repositories_changed': len(code_changes['repositories_processed']),
            'lines_changed': code_changes['lines_added'] + code_changes['lines_removed'],
            'impact_score': code_changes['impact_score']
        },
        'average_impact': {
            'commits_per_repo': round(code_changes['total_commits'] / len(code_changes['repositories_processed']),
                                      2),
            'lines_per_repo': round((code_changes['lines_added'] + code_changes['lines_removed']) /
                                    len(code_changes['repositories_processed']), 2)
        }
    }

    review_influence = {
        'total_reviews': reviews['total_reviews'],
        'unique_repos_reviewed': reviews['cross_repo_reviews']['unique_repositories'],
        'unique_authors_reviewed': reviews['cross_repo_reviews']['unique_authors'],
        'review_effectiveness': {
            'efficiency_score': reviews['cycle_analysis']['efficiency_score'],
            'single_review_success': reviews['cycle_analysis']['single_review_percentage']
        }
    }

    cross_repo_impact = {
        'total_repos': len(collaboration['cross_repo_impact']['repositories']),
        'impact_score': collaboration['cross_repo_impact']['impact_score'],
        'repository_spread': {
            'primary_languages': len(set(repo['primary_language']
                                         for repo in collaboration['cross_repo_impact']['review_activity'].values()
                                         if repo['primary_language'])),
            'active_repos': sum(1 for repo in collaboration['cross_repo_impact']['review_activity'].values()
                                if repo['reviews_given'] > 0 or repo['reviews_received'] > 0)
        }
    }

    return {
        'code_impact': code_impact,
        'review_influence': review_influence,
        'cross_repo_impact': cross_repo_impact,
        'impact_summary': {
            'overall_impact_score': _calculate_overall_impact_score(metrics),
            'primary_impact_areas': _identify_primary_impact_areas(metrics)
        }
    }


def _analyze_growth_trends(metrics: Dict[str, Any]) -> Dict[str, Any]:
    trends = {
        'code_velocity': _analyze_code_velocity_trend(metrics),
        'language_growth': _analyze_language_growth(metrics),
        'collaboration_growth': _analyze_collaboration_growth(metrics),
        'impact_growth': _analyze_impact_growth(metrics)
    }

    return trends


class InsightsGenerator:
    def __init__(self):
        self.percentile_benchmarks = {
            # Code Volume Metrics
            'commits_per_year': {
                95: 1000, # ~4/day - exceptional
                90: 500, # ~> 2/day - very good
                75: 250, # ~> 1/day - good
                50: 125 # ~1/2 day - average
            },
            'lines_changed': {95: 500000, 90: 250000, 75: 100000, 50: 50000},

            # PR Quality Metrics
            'pr_merge_rate': {
                95: 98,
                90: 90, # very few revisions needed
                75: 80, # some revisions normal
                50: 70 # multiple revisions common
            },
            'pr_size': {90: 200, 75: 400, 50: 800},  # Lines per PR (lower is better)

            # Review Metrics
            'review_engagement': {
                95: 300, # >1/day - exceptional engaged
                90: 200, # ~1/day - very engaged
                75: 100, # ~2/week - good engagement
                50: 50 # ~1/week - average engagement
            },
            'review_response_time': {
                90: 4, # half-day response
                75: 8, # same-day response
                50: 24 # next-day response
            },
            'review_efficiency': {90: 9.5, 75: 9.0, 50: 8.0},

            # Impact Metrics
            'code_impact': {95: 50000, 90: 10000, 75: 5000, 50: 1000},
            'repository_reach': {
                95: 20, # system-wide impact
                90: 10, # multi-team impact
                75: 5, # team-level impact
                50: 3 # focus area impact
            },

            # Collaboration Metrics
            'unique_collaborators': {
                95: 15, # cross-org collaboration
                90: 10, # multi-team collaboration
                75: 5, # full team collaboration
                50: 3 # core team collaboration
            },
            'cross_repo_impact': {95: 95, 90: 85, 75: 75, 50: 60},

            # Language Metrics
            'language_diversity': {
                90: 4, # polyglot developer
                75: 3, # multi-language developer
                50: 2  # specialist with backup
            },
            'expertise_depth': {90: 60, 75: 40, 50: 20},  # Percentage in primary language

            # Time Distribution
            'consistency_score': {90: 0.4, 75: 0.3, 50: 0.2},
            'active_days_per_week': {
                90: 5, # full-time contributions
                75: 4, # regular contributions
                50: 3 # periodic contributions
            },

            # Review Quality
            'single_review_success': {95: 95, 90: 90, 75: 80, 50: 70},
            'review_comment_ratio': {90: 2.0, 75: 1.5, 50: 1.0}  # Comments per review
        }

    def _calculate_developer_percentile(self, metrics: Dict[str, Any]) -> Dict[str, int]:
        code_changes = metrics['code_changes']
        reviews = metrics['code_reviews']

        percentiles = {
            'code_volume': _get_percentile_rank(
                code_changes['total_commits'],
                self.percentile_benchmarks['commits_per_year']
            ),
            'code_impact': _get_percentile_rank(
                code_changes['lines_added'] + code_changes['lines_removed'],
                self.percentile_benchmarks['lines_changed']
            ),
            'pr_quality': _get_percentile_rank(
                metrics['pull_requests']['merge_success_rate'],
                self.percentile_benchmarks['pr_merge_rate']
            ),
            'review_impact': _get_percentile_rank(
                reviews['total_reviews'],
                self.percentile_benchmarks['review_engagement']
            ),
            'repository_impact': _get_percentile_rank(
                len(metrics['collaboration']['cross_repo_impact']['repositories']),
                self.percentile_benchmarks['repository_reach']
            ),
            'collaboration': _get_percentile_rank(
                len(metrics['collaboration']['collaboration_network']['collaborators']),
                self.percentile_benchmarks['unique_collaborators']
            ),
            'review_efficiency': _get_percentile_rank(
                reviews['cycle_analysis']['efficiency_score'] * 10,
                self.percentile_benchmarks['review_efficiency']
            )
        }

        return percentiles

    def generate_insights(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'summary_insights': self._generate_summary_insights(metrics),
            'code_insights': _analyze_code_patterns(metrics),
            'workflow_insights': _analyze_workflow_patterns(metrics),
            'language_insights': _analyze_language_patterns(metrics),
            'collaboration_insights': _analyze_collaboration_patterns(metrics),
            'achievements': _generate_achievements(metrics),
            'growth_trends': _analyze_growth_trends(metrics),
            'recommendations': RecommendationsEngine().generate_recommendations(metrics),
        }

    def _generate_summary_insights(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        code_changes = metrics['code_changes']
        prs = metrics["pull_requests"]

        total_impact = _calculate_total_impact(metrics)
        percentile = self._calculate_developer_percentile(metrics)

        strength_areas = self._identify_strength_areas(metrics)
        growth_areas = self._identify_growth_areas(metrics)

        return {
            'impact_summary': {
                'total_impact': total_impact,
                'percentile': percentile,
                'highlight_metrics': {
                    'total_commits': code_changes['total_commits'],
                    'lines_changed': code_changes['lines_added'] + code_changes['lines_removed'],
                    'prs_merged': prs['merged_prs'],
                    'review_contributions': metrics['code_reviews']['total_reviews']
                }
            },
            'strength_areas': strength_areas,
            'growth_areas': growth_areas,
            'comparative_metrics': self._calculate_comparative_metrics(metrics)
        }

    def _identify_strength_areas(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        strengths = []

        commits = metrics['code_changes']['total_commits']
        if commits >= self.percentile_benchmarks['commits_per_year'][75]:
            percentile = _get_percentile_rank(commits, self.percentile_benchmarks['commits_per_year'])
            strengths.append({
                'area': 'Code Contribution',
                'level': 'Exceptional' if percentile >= 95 else 'Very High' if percentile >= 90 else 'High',
                'evidence': f"{commits} commits ({percentile}th percentile)",
                'percentile': percentile
            })

        pr_success_rate = metrics['pull_requests']['merge_success_rate']
        if pr_success_rate >= self.percentile_benchmarks['pr_merge_rate'][75]:
            percentile = _get_percentile_rank(pr_success_rate, self.percentile_benchmarks['pr_merge_rate'])
            strengths.append({
                'area': 'Code Quality',
                'level': 'Exceptional' if percentile >= 95 else 'Very High' if percentile >= 90 else 'High',
                'evidence': f"{pr_success_rate:.1f}% PR success rate ({percentile}th percentile)",
                'percentile': percentile
            })

        reviews = metrics['code_reviews']['total_reviews']
        if reviews >= self.percentile_benchmarks['review_engagement'][75]:
            percentile = _get_percentile_rank(reviews, self.percentile_benchmarks['review_engagement'])
            strengths.append({
                'area': 'Code Review',
                'level': 'Exceptional' if percentile >= 95 else 'Very High' if percentile >= 90 else 'High',
                'evidence': f"{reviews} reviews provided ({percentile}th percentile)",
                'percentile': percentile
            })

        repos = len(metrics['collaboration']['cross_repo_impact']['repositories'])
        if repos >= self.percentile_benchmarks['repository_reach'][75]:
            percentile = _get_percentile_rank(repos, self.percentile_benchmarks['repository_reach'])
            strengths.append({
                'area': 'System Impact',
                'level': 'Exceptional' if percentile >= 95 else 'Very High' if percentile >= 90 else 'High',
                'evidence': f"Impact across {repos} repositories ({percentile}th percentile)",
                'percentile': percentile
            })

        active_languages = len([
            lang for lang, pct in metrics['languages']['language_distribution'].items()
            if pct >= 5
        ])
        if active_languages >= self.percentile_benchmarks['language_diversity'][75]:
            percentile = _get_percentile_rank(active_languages, self.percentile_benchmarks['language_diversity'])
            strengths.append({
                'area': 'Technical Breadth',
                'level': 'Exceptional' if percentile >= 90 else 'High',
                'evidence': f"Proficient in {active_languages} languages ({percentile}th percentile)",
                'percentile': percentile
            })

        return sorted(strengths, key=lambda x: x['percentile'], reverse=True)

    def _identify_growth_areas(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        growth_areas = []

        # Review Response Time
        response_time = metrics['code_reviews']['response_time_avg'] * 24  # Convert to hours
        if response_time > self.percentile_benchmarks['review_response_time'][75]:
            target = self.percentile_benchmarks['review_response_time'][75]
            growth_areas.append({
                'area': 'Review Responsiveness',
                'current_metric': f"{response_time:.1f} hours",
                'target_metric': f"{target} hours",
                'improvement_potential': 'high' if response_time > target * 2 else 'medium'
            })

        avg_pr_size = (metrics['code_changes']['lines_added'] + metrics['code_changes']['lines_removed']) / \
                      metrics['pull_requests']['total_prs']
        if avg_pr_size > self.percentile_benchmarks['pr_size'][75]:
            target = self.percentile_benchmarks['pr_size'][75]
            growth_areas.append({
                'area': 'PR Size Optimization',
                'current_metric': f"{avg_pr_size:.0f} lines/PR",
                'target_metric': f"{target} lines/PR",
                'improvement_potential': 'high' if avg_pr_size > target * 2 else 'medium'
            })

        top_lang_usage = metrics['languages']['primary_languages'][0][1]
        if top_lang_usage < self.percentile_benchmarks['expertise_depth'][75]:
            target = self.percentile_benchmarks['expertise_depth'][75]
            growth_areas.append({
                'area': 'Technical Depth',
                'current_metric': f"{top_lang_usage:.1f}% in primary language",
                'target_metric': f"{target}% usage",
                'improvement_potential': 'medium'
            })

        consistency = metrics['code_changes']['commit_activity']['consistency_score']
        if consistency < self.percentile_benchmarks['consistency_score'][75]:
            target = self.percentile_benchmarks['consistency_score'][75]
            growth_areas.append({
                'area': 'Contribution Consistency',
                'current_metric': f"{consistency:.2f} consistency score",
                'target_metric': f"{target} score",
                'improvement_potential': 'medium'
            })

        return growth_areas

    def _calculate_comparative_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        code_changes = metrics['code_changes']
        reviews = metrics['code_reviews']
        prs = metrics['pull_requests']

        return {
            'velocity_metrics': {
                'commits': {
                    'value': code_changes['total_commits'],
                    'percentile': _get_percentile_rank(
                        code_changes['total_commits'],
                        self.percentile_benchmarks['commits_per_year']
                    ),
                    'industry_avg': self.percentile_benchmarks['commits_per_year'][50],
                    'industry_top': self.percentile_benchmarks['commits_per_year'][90]
                },
                'lines_changed': {
                    'value': code_changes['lines_added'] + code_changes['lines_removed'],
                    'percentile': _get_percentile_rank(
                        code_changes['lines_added'] + code_changes['lines_removed'],
                        self.percentile_benchmarks['lines_changed']
                    ),
                    'industry_avg': self.percentile_benchmarks['lines_changed'][50],
                    'industry_top': self.percentile_benchmarks['lines_changed'][90]
                }
            },
            'quality_metrics': {
                'pr_success_rate': {
                    'value': prs['merge_success_rate'],
                    'percentile': _get_percentile_rank(
                        prs['merge_success_rate'],
                        self.percentile_benchmarks['pr_merge_rate']
                    ),
                    'industry_avg': self.percentile_benchmarks['pr_merge_rate'][50],
                    'industry_top': self.percentile_benchmarks['pr_merge_rate'][90]
                },
                'review_efficiency': {
                    'value': reviews['cycle_analysis']['efficiency_score'],
                    'percentile': _get_percentile_rank(
                        reviews['cycle_analysis']['efficiency_score'] * 10,
                        self.percentile_benchmarks['review_efficiency']
                    ),
                    'industry_avg': self.percentile_benchmarks['review_efficiency'][50] / 10,
                    'industry_top': self.percentile_benchmarks['review_efficiency'][90] / 10
                }
            },
            'impact_metrics': {
                'repository_reach': {
                    'value': len(metrics['collaboration']['cross_repo_impact']['repositories']),
                    'percentile': _get_percentile_rank(
                        len(metrics['collaboration']['cross_repo_impact']['repositories']),
                        self.percentile_benchmarks['repository_reach']
                    ),
                    'industry_avg': self.percentile_benchmarks['repository_reach'][50],
                    'industry_top': self.percentile_benchmarks['repository_reach'][90]
                },
                'review_volume': {
                    'value': reviews['total_reviews'],
                    'percentile': _get_percentile_rank(
                        reviews['total_reviews'],
                        self.percentile_benchmarks['review_engagement']
                    ),
                    'industry_avg': self.percentile_benchmarks['review_engagement'][50],
                    'industry_top': self.percentile_benchmarks['review_engagement'][90]
                }
            },
            'collaboration_metrics': {
                'cross_team_impact': {
                    'value': metrics['collaboration']['cross_repo_impact']['impact_score'],
                    'percentile': _get_percentile_rank(
                        metrics['collaboration']['cross_repo_impact']['impact_score'],
                        self.percentile_benchmarks['cross_repo_impact']
                    ),
                    'industry_avg': self.percentile_benchmarks['cross_repo_impact'][50],
                    'industry_top': self.percentile_benchmarks['cross_repo_impact'][90]
                },
                'team_collaboration': {
                    'value': len(metrics['collaboration']['collaboration_network']['collaborators']),
                    'percentile': _get_percentile_rank(
                        len(metrics['collaboration']['collaboration_network']['collaborators']),
                        self.percentile_benchmarks['unique_collaborators']
                    ),
                    'industry_avg': self.percentile_benchmarks['unique_collaborators'][50],
                    'industry_top': self.percentile_benchmarks['unique_collaborators'][90]
                }
            },
            'percentile_summary': {
                'overall_percentile': self._calculate_weighted_percentile(metrics),
                'highest_percentile': self._identify_top_percentile(metrics),
                'areas_above_90th': self._count_high_percentile_areas(metrics)
            }
        }

    def _calculate_weighted_percentile(self, metrics: Dict[str, Any]) -> float:
        weights = {
            'commits': 0.20,
            'pr_quality': 0.20,
            'review_volume': 0.15,
            'repository_reach': 0.15,
            'review_efficiency': 0.15,
            'cross_team_impact': 0.15
        }

        scores = {
            'commits': _get_percentile_rank(
                metrics['code_changes']['total_commits'],
                self.percentile_benchmarks['commits_per_year']
            ),
            'pr_quality': _get_percentile_rank(
                metrics['pull_requests']['merge_success_rate'],
                self.percentile_benchmarks['pr_merge_rate']
            ),
            'review_volume': _get_percentile_rank(
                metrics['code_reviews']['total_reviews'],
                self.percentile_benchmarks['review_engagement']
            ),
            'repository_reach': _get_percentile_rank(
                len(metrics['collaboration']['cross_repo_impact']['repositories']),
                self.percentile_benchmarks['repository_reach']
            ),
            'review_efficiency': _get_percentile_rank(
                metrics['code_reviews']['cycle_analysis']['efficiency_score'] * 10,
                self.percentile_benchmarks['review_efficiency']
            ),
            'cross_team_impact': _get_percentile_rank(
                metrics['collaboration']['cross_repo_impact']['impact_score'],
                self.percentile_benchmarks['cross_repo_impact']
            )
        }

        return round(sum(scores[metric] * weight for metric, weight in weights.items()), 2)

    def _identify_top_percentile(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        percentiles = {
            'Code Volume': _get_percentile_rank(
                metrics['code_changes']['total_commits'],
                self.percentile_benchmarks['commits_per_year']
            ),
            'PR Quality': _get_percentile_rank(
                metrics['pull_requests']['merge_success_rate'],
                self.percentile_benchmarks['pr_merge_rate']
            ),
            'Review Impact': _get_percentile_rank(
                metrics['code_reviews']['total_reviews'],
                self.percentile_benchmarks['review_engagement']
            ),
            'Cross-Repo Impact': _get_percentile_rank(
                metrics['collaboration']['cross_repo_impact']['impact_score'],
                self.percentile_benchmarks['cross_repo_impact']
            )
        }

        top_area = max(percentiles.items(), key=lambda x: x[1])
        return {
            'area': top_area[0],
            'percentile': top_area[1]
        }

    def _count_high_percentile_areas(self, metrics: Dict[str, Any]) -> int:
        high_percentile_count = 0

        if metrics['code_changes']['total_commits'] >= self.percentile_benchmarks['commits_per_year'][90]:
            high_percentile_count += 1

        if metrics['pull_requests']['merge_success_rate'] >= self.percentile_benchmarks['pr_merge_rate'][90]:
            high_percentile_count += 1

        if metrics['code_reviews']['total_reviews'] >= self.percentile_benchmarks['review_engagement'][90]:
            high_percentile_count += 1

        if metrics['collaboration']['cross_repo_impact']['impact_score'] >= \
                self.percentile_benchmarks['cross_repo_impact'][90]:
            high_percentile_count += 1

        if len(metrics['collaboration']['cross_repo_impact']['repositories']) >= \
                self.percentile_benchmarks['repository_reach'][90]:
            high_percentile_count += 1

        return high_percentile_count


def _get_pattern_recommendations(metrics: Dict[str, Any], impact_pattern: str) -> List[Dict[str, Any]]:
    pattern_recommendations = {
        "EXCEPTIONAL_IMPACT": [
            {
                'category': 'Engineering Excellence',
                'title': 'Scale Your Impact',
                'description': (
                    f'Your exceptional output ({metrics["code_changes"]["total_commits"]} commits, '
                    f'{metrics["code_reviews"]["total_reviews"]} reviews) provides unique insights. '
                    f'Consider creating engineering guidelines or mentoring programs.'
                ),
                'metrics': {
                    'current': 'Top-tier contribution levels',
                    'next_step': 'Systemize and share your practices'
                },
                'priority': 'high'
            },
            {
                'category': 'Technical Strategy',
                'title': 'Architectural Leadership',
                'description': (
                    f'With impact across {len(metrics["code_changes"]["repositories_processed"])} repositories, '
                    f'you have unique insight into cross-cutting concerns. Consider driving architectural initiatives.'
                ),
                'metrics': {
                    'current': 'Broad system knowledge',
                    'next_step': 'Lead architectural improvements'
                },
                'priority': 'medium'
            }
        ],
        "HIGH_IMPACT": [
            {
                'category': 'Knowledge Sharing',
                'title': 'Amplify Team Impact',
                'description': (
                    f'Strong patterns in reviews ({metrics["code_reviews"]["total_reviews"]}) '
                    f'and PR success ({metrics["pull_requests"]["merge_success_rate"]:.1f}%). '
                    f'Share your expertise through documentation or workshops.'
                ),
                'metrics': {
                    'current': 'High review volume and PR success',
                    'next_step': 'Create team learning resources'
                },
                'priority': 'high'
            },
            {
                'category': 'Process Improvement',
                'title': 'Optimize Development Workflows',
                'description': 'Your experience across repositories positions you well to identify and improve development bottlenecks.',
                'metrics': {
                    'current': 'Cross-repo expertise',
                    'next_step': 'Propose workflow improvements'
                },
                'priority': 'medium'
            }
        ],
        "GROWING_IMPACT": [
            {
                'category': 'Impact Expansion',
                'title': 'Deepen Technical Impact',
                'description': (
                    f'Strong foundation with {metrics["code_changes"]["total_commits"]} commits. '
                    f'Consider taking on more complex features and architectural changes.'
                ),
                'metrics': {
                    'current': 'Solid contribution volume',
                    'next_step': 'Lead feature development'
                },
                'priority': 'high'
            },
            {
                'category': 'Review Leadership',
                'title': 'Expand Review Influence',
                'description': 'Your code quality insights are valuable. Increase your review participation across teams.',
                'metrics': {
                    'current_reviews': metrics["code_reviews"]["total_reviews"],
                    'target': f'{metrics["code_reviews"]["total_reviews"] * 1.5:.0f} reviews'
                },
                'priority': 'medium'
            }
        ],
        "BUILDING_IMPACT": [
            {
                'category': 'Contribution Growth',
                'title': 'Build Momentum',
                'description': (
                    f'Great start with {metrics["code_changes"]["total_commits"]} commits! '
                    f'Focus on regular contributions and expanding your repository knowledge.'
                ),
                'metrics': {
                    'current': 'Established contribution pattern',
                    'next_step': f'Target {metrics["code_changes"]["total_commits"] * 2} commits'
                },
                'priority': 'high'
            },
            {
                'category': 'Code Review Growth',
                'title': 'Strengthen Review Skills',
                'description': 'Build confidence through focused code reviews in your areas of expertise.',
                'metrics': {
                    'current_reviews': metrics["code_reviews"]["total_reviews"],
                    'next_milestone': f'{max(100, metrics["code_reviews"]["total_reviews"] * 2)} reviews'
                },
                'priority': 'medium'
            }
        ]
    }

    return pattern_recommendations[impact_pattern]


def _get_universal_recommendations(metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    recommendations = []

    # Large commit size recommendation
    avg_change_size = (metrics['code_changes']['lines_added'] + metrics['code_changes']['lines_removed']) / metrics['code_changes']['total_commits']
    if avg_change_size > 200:
        recommendations.append({
            'category': 'Code Quality',
            'title': 'Optimize Change Scope',
            'description': 'Consider breaking down larger changes for easier review and reduced merge conflicts.',
            'metrics': {
                'current_avg': f"{avg_change_size:.0f} lines/commit",
                'target': '100-200 lines per commit'
            },
            'priority': 'low'
        })

    # Review response time recommendation
    if metrics['code_reviews']['response_time_avg'] > 0.5:  # If average response time > 12 hours
        recommendations.append({
            'category': 'Collaboration',
            'title': 'Review Responsiveness',
            'description': 'Quick review feedback helps maintain team velocity.',
            'metrics': {
                'current_response': f"{metrics['code_reviews']['response_time_avg'] * 24:.1f} hours",
                'target': '< 12 hours'
            },
            'priority': 'low'
        })

    return recommendations


def _prioritize_recommendations(recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    priority_order = {'high': 0, 'medium': 1, 'low': 2}
    sorted_recs = sorted(recommendations,
                       key=lambda x: (priority_order[x['priority']], x['category']))
    return sorted_recs[:5]  # Return top 5 relevant recs...


class RecommendationsEngine:
    def __init__(self):
        self.impact_thresholds = {
            'commits': {
                'high': 1500,
                'medium': 750,
                'baseline': 250
            },
            'reviews': {
                'high': 400,
                'medium': 200,
                'baseline': 100
            },
            'repos': {
                'high': 20,
                'medium': 10,
                'baseline': 5
            },
            'pr_success': {
                'high': 95,
                'medium': 85,
                'baseline': 75
            }
        }

    def generate_recommendations(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        impact_pattern = self._analyze_impact_pattern(metrics)
        recommendations = []

        pattern_recs = _get_pattern_recommendations(metrics, impact_pattern)
        recommendations.extend(pattern_recs)

        universal_recs = _get_universal_recommendations(metrics)
        recommendations.extend(universal_recs)

        return _prioritize_recommendations(recommendations)

    def _analyze_impact_pattern(self, metrics: Dict[str, Any]) -> str:
        metrics_scores = []

        # Commit Impact (35%)
        commit_ratio = metrics['code_changes']['total_commits'] / self.impact_thresholds['commits']['high']
        metrics_scores.append(min(3.5, commit_ratio * 3.5))

        # Review Impact (25%)
        review_ratio = metrics['code_reviews']['total_reviews'] / self.impact_thresholds['reviews']['high']
        metrics_scores.append(min(2.5, review_ratio * 2.5))

        # Repository Breadth (20%)
        repo_ratio = len(metrics['code_changes']['repositories_processed']) / self.impact_thresholds['repos']['high']
        metrics_scores.append(min(2.0, repo_ratio * 2.0))

        # PR Quality (20%)
        pr_quality_ratio = metrics['pull_requests']['merge_success_rate'] / self.impact_thresholds['pr_success']['high']
        metrics_scores.append(min(2.0, pr_quality_ratio * 2.0))

        impact_score = sum(metrics_scores)

        if impact_score >= 8.5:
            return "EXCEPTIONAL_IMPACT"
        elif impact_score >= 7:
            return "HIGH_IMPACT"
        elif impact_score >= 5:
            return "GROWING_IMPACT"
        return "BUILDING_IMPACT"
