import pandas as pd
import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional
import hashlib

class CollaborationManager:
    """Team collaboration features for shared data analysis projects."""
    
    def __init__(self, db_manager=None):
        self.db_manager = db_manager
        self.collaboration_data = {}
        
    def create_project(self, project_name: str, description: str, 
                      created_by: str, dataset_id: Optional[int] = None) -> Dict[str, Any]:
        """Create a new collaborative project."""
        try:
            project_id = hashlib.md5(f"{project_name}_{datetime.now()}".encode()).hexdigest()[:8]
            
            project_data = {
                'project_id': project_id,
                'name': project_name,
                'description': description,
                'created_by': created_by,
                'created_at': datetime.now().isoformat(),
                'dataset_id': dataset_id,
                'team_members': [created_by],
                'analysis_history': [],
                'shared_insights': [],
                'project_status': 'active',
                'permissions': {
                    created_by: 'owner'
                }
            }
            
            if self.db_manager:
                self._save_project_to_db(project_data)
            
            return {
                'success': True,
                'project_id': project_id,
                'message': f"Project '{project_name}' created successfully",
                'project_data': project_data
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Failed to create project: {str(e)}"
            }
    
    def add_team_member(self, project_id: str, member_name: str, 
                       permission_level: str = 'viewer', added_by: str = 'system') -> Dict[str, Any]:
        """Add a team member to a collaborative project."""
        try:
            project_data = self._get_project_data(project_id)
            if not project_data:
                return {'success': False, 'error': 'Project not found'}
            
            if member_name not in project_data['team_members']:
                project_data['team_members'].append(member_name)
                project_data['permissions'][member_name] = permission_level
                
                # Add activity log
                activity = {
                    'timestamp': datetime.now().isoformat(),
                    'action': 'member_added',
                    'user': added_by,
                    'details': f"Added {member_name} as {permission_level}"
                }
                project_data['analysis_history'].append(activity)
                
                if self.db_manager:
                    self._update_project_in_db(project_data)
                
                return {
                    'success': True,
                    'message': f"{member_name} added to project as {permission_level}"
                }
            else:
                return {
                    'success': False,
                    'error': f"{member_name} is already a team member"
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': f"Failed to add team member: {str(e)}"
            }
    
    def log_analysis_activity(self, project_id: str, user: str, analysis_type: str, 
                             results: Dict[str, Any], notes: str = "") -> Dict[str, Any]:
        """Log analysis activity for team collaboration."""
        try:
            project_data = self._get_project_data(project_id)
            if not project_data:
                return {'success': False, 'error': 'Project not found'}
            
            activity = {
                'timestamp': datetime.now().isoformat(),
                'user': user,
                'analysis_type': analysis_type,
                'results_summary': self._summarize_results(results),
                'notes': notes,
                'full_results_id': self._store_full_results(project_id, results)
            }
            
            project_data['analysis_history'].append(activity)
            
            if self.db_manager:
                self._update_project_in_db(project_data)
            
            return {
                'success': True,
                'message': f"Analysis activity logged for {user}",
                'activity_id': len(project_data['analysis_history']) - 1
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Failed to log activity: {str(e)}"
            }
    
    def share_insight(self, project_id: str, user: str, insight_title: str, 
                     insight_content: str, insight_type: str = 'general') -> Dict[str, Any]:
        """Share an insight with the team."""
        try:
            project_data = self._get_project_data(project_id)
            if not project_data:
                return {'success': False, 'error': 'Project not found'}
            
            insight = {
                'insight_id': len(project_data['shared_insights']),
                'timestamp': datetime.now().isoformat(),
                'user': user,
                'title': insight_title,
                'content': insight_content,
                'type': insight_type,
                'comments': [],
                'likes': 0
            }
            
            project_data['shared_insights'].append(insight)
            
            if self.db_manager:
                self._update_project_in_db(project_data)
            
            return {
                'success': True,
                'message': f"Insight '{insight_title}' shared successfully",
                'insight_id': insight['insight_id']
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Failed to share insight: {str(e)}"
            }
    
    def add_comment_to_insight(self, project_id: str, insight_id: int, 
                              user: str, comment: str) -> Dict[str, Any]:
        """Add a comment to a shared insight."""
        try:
            project_data = self._get_project_data(project_id)
            if not project_data:
                return {'success': False, 'error': 'Project not found'}
            
            if insight_id >= len(project_data['shared_insights']):
                return {'success': False, 'error': 'Insight not found'}
            
            comment_data = {
                'timestamp': datetime.now().isoformat(),
                'user': user,
                'comment': comment
            }
            
            project_data['shared_insights'][insight_id]['comments'].append(comment_data)
            
            if self.db_manager:
                self._update_project_in_db(project_data)
            
            return {
                'success': True,
                'message': 'Comment added successfully'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Failed to add comment: {str(e)}"
            }
    
    def get_project_summary(self, project_id: str) -> Dict[str, Any]:
        """Get comprehensive project summary for team dashboard."""
        try:
            project_data = self._get_project_data(project_id)
            if not project_data:
                return {'error': 'Project not found'}
            
            # Calculate project statistics
            total_analyses = len(project_data['analysis_history'])
            total_insights = len(project_data['shared_insights'])
            team_size = len(project_data['team_members'])
            
            # Get recent activity
            recent_activity = sorted(
                project_data['analysis_history'][-10:], 
                key=lambda x: x['timestamp'], 
                reverse=True
            )
            
            # Get top contributors
            user_activity = {}
            for activity in project_data['analysis_history']:
                user = activity['user']
                user_activity[user] = user_activity.get(user, 0) + 1
            
            top_contributors = sorted(
                user_activity.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5]
            
            # Analysis type distribution
            analysis_types = {}
            for activity in project_data['analysis_history']:
                analysis_type = activity['analysis_type']
                analysis_types[analysis_type] = analysis_types.get(analysis_type, 0) + 1
            
            return {
                'project_info': {
                    'name': project_data['name'],
                    'description': project_data['description'],
                    'created_by': project_data['created_by'],
                    'created_at': project_data['created_at'],
                    'status': project_data['project_status']
                },
                'statistics': {
                    'team_size': team_size,
                    'total_analyses': total_analyses,
                    'total_insights': total_insights,
                    'dataset_id': project_data.get('dataset_id')
                },
                'team_members': project_data['team_members'],
                'recent_activity': recent_activity,
                'top_contributors': top_contributors,
                'analysis_distribution': analysis_types,
                'shared_insights_preview': project_data['shared_insights'][-5:] if project_data['shared_insights'] else []
            }
            
        except Exception as e:
            return {'error': f"Failed to get project summary: {str(e)}"}
    
    def get_team_analytics(self, project_id: str) -> Dict[str, Any]:
        """Get team collaboration analytics."""
        try:
            project_data = self._get_project_data(project_id)
            if not project_data:
                return {'error': 'Project not found'}
            
            # Time-based activity analysis
            daily_activity = {}
            user_contributions = {}
            
            for activity in project_data['analysis_history']:
                # Daily activity
                date = activity['timestamp'].split('T')[0]
                daily_activity[date] = daily_activity.get(date, 0) + 1
                
                # User contributions
                user = activity['user']
                if user not in user_contributions:
                    user_contributions[user] = {
                        'total_analyses': 0,
                        'analysis_types': {},
                        'first_contribution': activity['timestamp'],
                        'last_contribution': activity['timestamp']
                    }
                
                user_contributions[user]['total_analyses'] += 1
                analysis_type = activity['analysis_type']
                user_contributions[user]['analysis_types'][analysis_type] = \
                    user_contributions[user]['analysis_types'].get(analysis_type, 0) + 1
                user_contributions[user]['last_contribution'] = activity['timestamp']
            
            # Collaboration patterns
            collaboration_score = self._calculate_collaboration_score(project_data)
            
            return {
                'daily_activity': daily_activity,
                'user_contributions': user_contributions,
                'collaboration_metrics': {
                    'collaboration_score': collaboration_score,
                    'avg_analyses_per_user': total_analyses / team_size if team_size > 0 else 0,
                    'insight_to_analysis_ratio': total_insights / total_analyses if total_analyses > 0 else 0
                },
                'engagement_metrics': {
                    'active_users_last_week': self._count_active_users_last_week(project_data),
                    'most_popular_analysis_type': max(analysis_types.items(), key=lambda x: x[1])[0] if analysis_types else None
                }
            }
            
        except Exception as e:
            return {'error': f"Failed to get team analytics: {str(e)}"}
    
    def export_project_report(self, project_id: str) -> Dict[str, Any]:
        """Export comprehensive project report for team review."""
        try:
            project_summary = self.get_project_summary(project_id)
            team_analytics = self.get_team_analytics(project_id)
            
            if 'error' in project_summary or 'error' in team_analytics:
                return {'error': 'Failed to generate report'}
            
            report = {
                'report_generated_at': datetime.now().isoformat(),
                'project_summary': project_summary,
                'team_analytics': team_analytics,
                'recommendations': self._generate_team_recommendations(project_summary, team_analytics)
            }
            
            return {
                'success': True,
                'report': report,
                'message': 'Project report generated successfully'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Failed to export report: {str(e)}"
            }
    
    def _get_project_data(self, project_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve project data from storage."""
        if self.db_manager:
            return self._get_project_from_db(project_id)
        else:
            return self.collaboration_data.get(project_id)
    
    def _save_project_to_db(self, project_data: Dict[str, Any]):
        """Save project data to database (simplified implementation)."""
        # This would integrate with the actual database
        # For now, store in memory
        self.collaboration_data[project_data['project_id']] = project_data
    
    def _update_project_in_db(self, project_data: Dict[str, Any]):
        """Update project data in database."""
        self.collaboration_data[project_data['project_id']] = project_data
    
    def _get_project_from_db(self, project_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve project data from database."""
        return self.collaboration_data.get(project_id)
    
    def _summarize_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of analysis results for activity log."""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'result_type': 'analysis_summary'
        }
        
        # Extract key metrics based on result type
        if 'model_performance' in results:
            # ML results
            summary['best_model'] = results.get('best_model', 'Unknown')
            summary['metrics_summary'] = 'ML model evaluation completed'
        elif 'missing_percentage' in results:
            # Data profiling results
            summary['data_quality'] = f"{results.get('missing_percentage', 0):.1f}% missing data"
            summary['metrics_summary'] = 'Data profiling completed'
        else:
            summary['metrics_summary'] = 'Analysis completed'
        
        return summary
    
    def _store_full_results(self, project_id: str, results: Dict[str, Any]) -> str:
        """Store full analysis results and return storage ID."""
        # In a real implementation, this would store results in database
        # For now, return a simple ID
        return f"{project_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def _calculate_collaboration_score(self, project_data: Dict[str, Any]) -> float:
        """Calculate collaboration score based on team activity patterns."""
        team_size = len(project_data['team_members'])
        total_analyses = len(project_data['analysis_history'])
        total_insights = len(project_data['shared_insights'])
        
        if team_size <= 1:
            return 0.0
        
        # Base score on activity distribution
        user_activity = {}
        for activity in project_data['analysis_history']:
            user = activity['user']
            user_activity[user] = user_activity.get(user, 0) + 1
        
        # Calculate activity distribution evenness
        if not user_activity:
            return 0.0
        
        activity_values = list(user_activity.values())
        mean_activity = sum(activity_values) / len(activity_values)
        activity_variance = sum((x - mean_activity) ** 2 for x in activity_values) / len(activity_values)
        
        # Lower variance = better collaboration
        collaboration_score = max(0, 100 - (activity_variance / mean_activity * 10)) if mean_activity > 0 else 0
        
        # Bonus for insights sharing
        insight_bonus = min(20, total_insights * 2)
        
        return min(100, collaboration_score + insight_bonus)
    
    def _count_active_users_last_week(self, project_data: Dict[str, Any]) -> int:
        """Count users active in the last week."""
        from datetime import datetime, timedelta
        
        week_ago = datetime.now() - timedelta(days=7)
        active_users = set()
        
        for activity in project_data['analysis_history']:
            activity_time = datetime.fromisoformat(activity['timestamp'].replace('Z', '+00:00'))
            if activity_time > week_ago:
                active_users.add(activity['user'])
        
        return len(active_users)
    
    def _generate_team_recommendations(self, project_summary: Dict[str, Any], 
                                     team_analytics: Dict[str, Any]) -> List[str]:
        """Generate recommendations for team collaboration improvement."""
        recommendations = []
        
        # Check team engagement
        team_size = project_summary['statistics']['team_size']
        total_analyses = project_summary['statistics']['total_analyses']
        
        if team_size > 1 and total_analyses / team_size < 2:
            recommendations.append("Encourage more team members to contribute analyses")
        
        # Check collaboration score
        collaboration_score = team_analytics['collaboration_metrics']['collaboration_score']
        if collaboration_score < 50:
            recommendations.append("Improve collaboration by encouraging knowledge sharing")
        
        # Check insight sharing
        insight_ratio = team_analytics['collaboration_metrics']['insight_to_analysis_ratio']
        if insight_ratio < 0.2:
            recommendations.append("Team members should share more insights from their analyses")
        
        # Check activity distribution
        user_contributions = team_analytics['user_contributions']
        if len(user_contributions) > 1:
            contributions = [data['total_analyses'] for data in user_contributions.values()]
            if max(contributions) > 3 * min(contributions):
                recommendations.append("Balance workload more evenly among team members")
        
        if not recommendations:
            recommendations.append("Team collaboration is working well - keep up the good work!")
        
        return recommendations