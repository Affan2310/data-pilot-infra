import os
import pandas as pd
import psycopg2
from sqlalchemy import create_engine, text
from datetime import datetime
from typing import Dict, Any, Optional, List
import json

class DatabaseManager:
    """Manages database operations for DataSciPilot application."""
    
    def __init__(self):
        self.database_url = os.getenv('DATABASE_URL')
        if not self.database_url:
            raise Exception("DATABASE_URL environment variable not found")
        
        self.engine = create_engine(self.database_url)
        self.init_database()
    
    def init_database(self):
        """Initialize database tables."""
        try:
            with self.engine.connect() as conn:
                # Create datasets table
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS datasets (
                        id SERIAL PRIMARY KEY,
                        name VARCHAR(255) NOT NULL,
                        upload_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        rows INTEGER,
                        columns INTEGER,
                        file_type VARCHAR(50),
                        memory_usage_mb FLOAT,
                        missing_percentage FLOAT,
                        duplicates INTEGER,
                        profile_results TEXT,
                        ml_results TEXT,
                        ai_insights TEXT
                    )
                """))
                
                # Create analysis sessions table
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS analysis_sessions (
                        id SERIAL PRIMARY KEY,
                        dataset_id INTEGER,
                        session_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        analysis_type VARCHAR(100),
                        results TEXT,
                        execution_time_seconds FLOAT
                    )
                """))
                
                conn.commit()
                print("Database tables initialized successfully")
        except Exception as e:
            print(f"Error initializing database: {e}")
    
    def save_dataset(self, name: str, df: pd.DataFrame, file_type: str = 'csv') -> int:
        """Save dataset metadata to database and return dataset ID."""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                    INSERT INTO datasets (name, rows, columns, file_type, memory_usage_mb, missing_percentage, duplicates)
                    VALUES (:name, :rows, :columns, :file_type, :memory_usage_mb, :missing_percentage, :duplicates)
                    RETURNING id
                """), {
                    'name': name,
                    'rows': len(df),
                    'columns': len(df.columns),
                    'file_type': file_type,
                    'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024),
                    'missing_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
                    'duplicates': int(df.duplicated().sum())
                })
                
                dataset_id = result.fetchone()[0]
                conn.commit()
                
                # Save the actual data to a separate table
                self._save_dataset_data(df, dataset_id, name)
                
                return dataset_id
        except Exception as e:
            raise Exception(f"Error saving dataset: {e}")
    
    def _save_dataset_data(self, df: pd.DataFrame, dataset_id: int, table_name: str):
        """Save actual dataset data to database."""
        try:
            # Clean table name for SQL
            clean_table_name = f"dataset_{dataset_id}_{table_name.lower().replace(' ', '_').replace('-', '_')}"
            clean_table_name = ''.join(c for c in clean_table_name if c.isalnum() or c == '_')[:63]  # PostgreSQL limit
            
            # Save dataframe to database
            df.to_sql(clean_table_name, self.engine, if_exists='replace', index=False)
            print(f"Dataset saved to table: {clean_table_name}")
        except Exception as e:
            print(f"Error saving dataset data: {e}")
    
    def update_dataset_analysis(self, dataset_id: int, analysis_type: str, results: Dict[str, Any]):
        """Update dataset with analysis results."""
        try:
            results_json = json.dumps(results, default=str)
            
            with self.engine.connect() as conn:
                if analysis_type == 'profiling':
                    conn.execute(text("""
                        UPDATE datasets SET profile_results = :results WHERE id = :id
                    """), {'results': results_json, 'id': dataset_id})
                elif analysis_type == 'ml':
                    conn.execute(text("""
                        UPDATE datasets SET ml_results = :results WHERE id = :id
                    """), {'results': results_json, 'id': dataset_id})
                elif analysis_type == 'ai_insights':
                    conn.execute(text("""
                        UPDATE datasets SET ai_insights = :results WHERE id = :id
                    """), {'results': results_json, 'id': dataset_id})
                
                conn.commit()
                
                # Also save to analysis session
                self._save_analysis_session(dataset_id, analysis_type, results)
                
        except Exception as e:
            raise Exception(f"Error updating dataset analysis: {e}")
    
    def _save_analysis_session(self, dataset_id: int, analysis_type: str, results: Dict[str, Any]):
        """Save analysis session to history."""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("""
                    INSERT INTO analysis_sessions (dataset_id, analysis_type, results)
                    VALUES (:dataset_id, :analysis_type, :results)
                """), {
                    'dataset_id': dataset_id,
                    'analysis_type': analysis_type,
                    'results': json.dumps(results, default=str)
                })
                conn.commit()
        except Exception as e:
            print(f"Error saving analysis session: {e}")
    
    def get_datasets_history(self) -> List[Dict[str, Any]]:
        """Get list of all analyzed datasets."""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT id, name, upload_timestamp, rows, columns, file_type, memory_usage_mb, 
                           missing_percentage, duplicates, 
                           (profile_results IS NOT NULL) as has_profiling,
                           (ml_results IS NOT NULL) as has_ml_results,
                           (ai_insights IS NOT NULL) as has_ai_insights
                    FROM datasets 
                    ORDER BY upload_timestamp DESC
                """))
                
                history = []
                for row in result:
                    history.append({
                        'id': row[0],
                        'name': row[1],
                        'upload_timestamp': row[2].isoformat() if row[2] else None,
                        'rows': row[3],
                        'columns': row[4],
                        'file_type': row[5],
                        'memory_usage_mb': row[6],
                        'missing_percentage': row[7],
                        'duplicates': row[8],
                        'has_profiling': row[9],
                        'has_ml_results': row[10],
                        'has_ai_insights': row[11]
                    })
                
                return history
        except Exception as e:
            raise Exception(f"Error getting datasets history: {e}")
    
    def get_dataset_details(self, dataset_id: int) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific dataset."""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT id, name, upload_timestamp, rows, columns, file_type, 
                           memory_usage_mb, missing_percentage, duplicates,
                           profile_results, ml_results, ai_insights
                    FROM datasets WHERE id = :id
                """), {'id': dataset_id})
                
                row = result.fetchone()
                if not row:
                    return None
                
                details = {
                    'id': row[0],
                    'name': row[1],
                    'upload_timestamp': row[2].isoformat() if row[2] else None,
                    'rows': row[3],
                    'columns': row[4],
                    'file_type': row[5],
                    'memory_usage_mb': row[6],
                    'missing_percentage': row[7],
                    'duplicates': row[8]
                }
                
                # Parse JSON results
                if row[9]:  # profile_results
                    details['profile_results'] = json.loads(row[9])
                
                if row[10]:  # ml_results
                    details['ml_results'] = json.loads(row[10])
                
                if row[11]:  # ai_insights
                    details['ai_insights'] = row[11]
                
                return details
        except Exception as e:
            raise Exception(f"Error getting dataset details: {e}")
    
    def load_dataset_from_db(self, dataset_id: int) -> Optional[pd.DataFrame]:
        """Load dataset data from database."""
        try:
            with self.engine.connect() as conn:
                # Get dataset name
                result = conn.execute(text("SELECT name FROM datasets WHERE id = :id"), {'id': dataset_id})
                row = result.fetchone()
                if not row:
                    return None
                
                dataset_name = row[0]
                
                # Construct table name
                clean_table_name = f"dataset_{dataset_id}_{dataset_name.lower().replace(' ', '_').replace('-', '_')}"
                clean_table_name = ''.join(c for c in clean_table_name if c.isalnum() or c == '_')[:63]
                
                # Load data
                df = pd.read_sql_table(clean_table_name, self.engine)
                return df
                
        except Exception as e:
            print(f"Error loading dataset from database: {e}")
            return None
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get overall database statistics."""
        try:
            with self.engine.connect() as conn:
                # Total datasets
                result = conn.execute(text("SELECT COUNT(*) FROM datasets"))
                total_datasets = result.fetchone()[0]
                
                # Total analyses
                result = conn.execute(text("SELECT COUNT(*) FROM analysis_sessions"))
                total_analyses = result.fetchone()[0]
                
                # Recent activity (today)
                result = conn.execute(text("""
                    SELECT COUNT(*) FROM datasets 
                    WHERE upload_timestamp >= CURRENT_DATE
                """))
                datasets_today = result.fetchone()[0]
                
                result = conn.execute(text("""
                    SELECT COUNT(*) FROM analysis_sessions 
                    WHERE session_timestamp >= CURRENT_DATE
                """))
                analyses_today = result.fetchone()[0]
                
                return {
                    'total_datasets': total_datasets,
                    'total_analyses': total_analyses,
                    'datasets_today': datasets_today,
                    'analyses_today': analyses_today
                }
        except Exception as e:
            raise Exception(f"Error getting database stats: {e}")
    
    def delete_dataset(self, dataset_id: int) -> bool:
        """Delete a dataset and all associated data."""
        try:
            with self.engine.connect() as conn:
                # Get dataset name for table cleanup
                result = conn.execute(text("SELECT name FROM datasets WHERE id = :id"), {'id': dataset_id})
                row = result.fetchone()
                if not row:
                    return False
                
                dataset_name = row[0]
                
                # Delete associated analysis sessions
                conn.execute(text("DELETE FROM analysis_sessions WHERE dataset_id = :id"), {'id': dataset_id})
                
                # Delete dataset record
                conn.execute(text("DELETE FROM datasets WHERE id = :id"), {'id': dataset_id})
                
                # Delete dataset table
                clean_table_name = f"dataset_{dataset_id}_{dataset_name.lower().replace(' ', '_').replace('-', '_')}"
                clean_table_name = ''.join(c for c in clean_table_name if c.isalnum() or c == '_')[:63]
                
                try:
                    conn.execute(text(f"DROP TABLE IF EXISTS {clean_table_name}"))
                except Exception as e:
                    print(f"Error dropping dataset table: {e}")
                
                conn.commit()
                return True
                
        except Exception as e:
            raise Exception(f"Error deleting dataset: {e}")