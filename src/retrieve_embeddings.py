#!/usr/bin/env python3
"""
Universal Embeddings Retrieval Script for pgVector PostgreSQL Database

This script provides a unified interface to retrieve embeddings from your pgVector database
with various filtering options including user, table, date range, and similarity search.

Usage:
    python retrieve_embeddings.py --user ejeannot --table text --limit 10
    python retrieve_embeddings.py --user ejeannot --experiment "text_and_images" --similarity "apple fruit"
    python retrieve_embeddings.py --list-databases
    python retrieve_embeddings.py --list-tables --database "text_and_images_ejeannot"
"""

import argparse
import sys
import os
import json
import psycopg2
from psycopg2 import sql
import numpy as np
from datetime import datetime, date
from typing import List, Dict, Tuple, Optional
import ast

# Optional: Import for similarity search (install with: pip install sentence-transformers)
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers not available. Similarity search will be disabled.")


class EmbeddingsRetriever:
    """Main class for retrieving embeddings from pgVector databases."""

    def __init__(self, host="localhost", port="5432", superuser="postgres", password="postgres"):
        self.host = host
        self.port = port
        self.superuser = superuser
        self.password = password
        self.connection = None
        self.cursor = None

    def connect(self, database="postgres", user=None, password=None):
        """Connect to PostgreSQL database."""
        try:
            user = user or self.superuser
            password = password or self.password
            self.connection = psycopg2.connect(
                dbname=database,
                user=user,
                password=password,
                host=self.host,
                port=self.port
            )
            self.cursor = self.connection.cursor()
            print(f"✅ Connected to database '{database}' as user '{user}' on {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"❌ Failed to connect to database: {e}")
            return False

    def disconnect(self):
        """Close database connection."""
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()

    def list_databases(self, username_filter=None):
        """List all databases, optionally filtered by username."""
        if not self.connect():
            return []

        query = "SELECT datname FROM pg_database WHERE datistemplate = false"
        params = []

        if username_filter:
            query += " AND datname LIKE %s"
            params.append(f"%{username_filter}%")

        query += " ORDER BY datname"

        try:
            self.cursor.execute(query, params)
            databases = [row[0] for row in self.cursor.fetchall()]

            print(f"\n📋 Available databases:")
            for db in databases:
                print(f"  • {db}")

            return databases
        except Exception as e:
            print(f"❌ Error listing databases: {e}")
            return []

    def list_tables(self, database):
        """List all tables in a database with vector columns."""
        if not self.connect(database):
            return []

        # Check if pgvector extension is available
        self.cursor.execute("SELECT count(*) FROM pg_extension WHERE extname = 'vector'")
        if self.cursor.fetchone()[0] == 0:
            print(f"⚠️  pgvector extension not enabled in database '{database}'")
            return []

        # Find tables with vector columns
        query = """
        SELECT t.table_name, c.column_name, c.data_type, c.character_maximum_length
        FROM information_schema.tables t
        JOIN information_schema.columns c ON t.table_name = c.table_name
        WHERE t.table_schema = 'public'
        AND c.data_type = 'USER-DEFINED'
        AND EXISTS (
            SELECT 1 FROM pg_type pt
            WHERE pt.typname = 'vector'
            AND c.udt_name = pt.typname
        )
        ORDER BY t.table_name, c.column_name
        """

        try:
            self.cursor.execute(query)
            results = self.cursor.fetchall()

            if not results:
                print(f"\n📋 No tables with vector columns found in database '{database}'")
                return []

            print(f"\n📋 Tables with vector columns in database '{database}':")
            current_table = None
            tables = []

            for table_name, column_name, data_type, max_length in results:
                if table_name != current_table:
                    print(f"  📊 {table_name}")
                    current_table = table_name
                    tables.append(table_name)

                # Get vector dimension
                vector_dim = self.get_vector_dimension(table_name, column_name)
                print(f"    └── {column_name}: VECTOR({vector_dim})")

            return tables
        except Exception as e:
            print(f"❌ Error listing tables: {e}")
            return []

    def get_vector_dimension(self, table_name, column_name):
        """Get the dimension of a vector column."""
        try:
            query = """
            SELECT atttypmod
            FROM pg_attribute
            WHERE attrelid = %s::regclass AND attname = %s
            """
            self.cursor.execute(query, (table_name, column_name))
            result = self.cursor.fetchone()
            return result[0] if result else "unknown"
        except:
            return "unknown"

    def build_database_name(self, username, experiment):
        """Build database name following the project's naming convention."""
        db_name = f"{experiment}_{username}".replace(" ", "_").lower()
        return db_name

    def retrieve_embeddings(self,
                          username: str,
                          experiment: str = "text_and_images",
                          table: str = None,
                          limit: int = None,
                          date_from: str = None,
                          date_to: str = None,
                          s3_object_filter: str = None,
                          similarity_query: str = None,
                          similarity_model: str = "all-MiniLM-L6-v2",
                          top_k: int = 10,
                          output_format: str = "table",
                          save_to_file: str = None) -> List[Dict]:
        """
        Retrieve embeddings with various filtering options.

        Args:
            username: Database username
            experiment: Experiment name (used to build database name)
            table: Specific table name
            limit: Maximum number of records to retrieve
            date_from: Start date filter (YYYY-MM-DD)
            date_to: End date filter (YYYY-MM-DD)
            s3_object_filter: Filter by S3 object name (supports wildcards)
            similarity_query: Text for similarity search
            similarity_model: Model for similarity search
            top_k: Number of similar results
            output_format: Output format ('table', 'json', 'csv')
            save_to_file: Save results to file

        Returns:
            List of dictionaries containing the results
        """

        # Build database name
        database = self.build_database_name(username, experiment)

        # For Docker environment, try connecting with ejeannot user first, then postgres
        connected = False
        if not connected:
            connected = self.connect(database, username, "ejeannot")
        if not connected:
            connected = self.connect(database, "ejeannot", "ejeannot")
        if not connected:
            connected = self.connect(database, "postgres", "postgres")

        if not connected:
            return []

        try:
            # If no table specified, find tables with embeddings
            if not table:
                tables = self.list_tables(database)
                if not tables:
                    print("❌ No tables with embeddings found")
                    return []
                table = tables[0]  # Use first table by default
                print(f"📊 Using table: {table}")

            # Build the query
            query_parts = []
            params = []

            # Base SELECT
            base_query = f"SELECT * FROM {table}"

            # Handle similarity search
            if similarity_query and SENTENCE_TRANSFORMERS_AVAILABLE:
                results = self._similarity_search(table, similarity_query, similarity_model, top_k)
                return self._format_output(results, output_format, save_to_file)

            # Build WHERE conditions
            where_conditions = []

            if date_from:
                where_conditions.append("date >= %s")
                params.append(date_from)

            if date_to:
                where_conditions.append("date <= %s")
                params.append(date_to)

            if s3_object_filter:
                where_conditions.append("s3_object LIKE %s")
                params.append(f"%{s3_object_filter}%")

            # Combine query parts
            if where_conditions:
                base_query += " WHERE " + " AND ".join(where_conditions)

            base_query += " ORDER BY id"

            if limit:
                base_query += f" LIMIT {limit}"

            print(f"🔍 Executing query: {base_query}")
            self.cursor.execute(base_query, params)

            # Get column names
            column_names = [desc[0] for desc in self.cursor.description]

            # Fetch results
            rows = self.cursor.fetchall()

            if not rows:
                print("📭 No results found")
                return []

            print(f"✅ Found {len(rows)} records")

            # Convert to list of dictionaries
            results = []
            for row in rows:
                record = dict(zip(column_names, row))
                # Convert embedding to list if it's a vector
                if 'embedding' in record and record['embedding'] is not None:
                    if isinstance(record['embedding'], str):
                        try:
                            record['embedding'] = ast.literal_eval(record['embedding'])
                        except:
                            pass
                results.append(record)

            return self._format_output(results, output_format, save_to_file)

        except Exception as e:
            print(f"❌ Error retrieving embeddings: {e}")
            return []

    def _similarity_search(self, table, query_text, model_name, top_k):
        """Perform similarity search using sentence transformers."""
        try:
            print(f"🔍 Performing similarity search for: '{query_text}'")

            # Load model
            model = SentenceTransformer(model_name)

            # Generate query embedding
            query_embedding = model.encode(query_text).tolist()

            # Perform similarity search
            query = f"""
            SELECT *, embedding <-> %s::vector as distance
            FROM {table}
            WHERE embedding IS NOT NULL
            ORDER BY embedding <-> %s::vector
            LIMIT %s
            """

            self.cursor.execute(query, (query_embedding, query_embedding, top_k))

            # Get column names
            column_names = [desc[0] for desc in self.cursor.description]

            # Fetch results
            rows = self.cursor.fetchall()

            results = []
            for row in rows:
                record = dict(zip(column_names, row))
                # Convert embedding to list if it's a vector
                if 'embedding' in record and record['embedding'] is not None:
                    if isinstance(record['embedding'], str):
                        try:
                            record['embedding'] = ast.literal_eval(record['embedding'])
                        except:
                            pass
                results.append(record)

            print(f"✅ Found {len(results)} similar records")
            return results

        except Exception as e:
            print(f"❌ Error in similarity search: {e}")
            return []

    def _format_output(self, results, output_format, save_to_file):
        """Format and optionally save the output."""
        if not results:
            return results

        if output_format == "json":
            output = json.dumps(results, indent=2, default=str)
            print(output)
        elif output_format == "csv":
            self._print_csv(results)
        else:  # table format
            self._print_table(results)

        if save_to_file:
            self._save_to_file(results, save_to_file, output_format)

        return results

    def _print_table(self, results):
        """Print results in a formatted table."""
        if not results:
            return

        # Get all column names
        columns = list(results[0].keys())

        # Calculate column widths
        col_widths = {}
        for col in columns:
            if col == 'embedding':
                col_widths[col] = 20  # Truncate embeddings
            else:
                max_width = max(len(str(col)),
                              max(len(str(row.get(col, ''))) for row in results[:10]))
                col_widths[col] = min(max_width, 50)  # Cap at 50 chars

        # Print header
        header = " | ".join(col.ljust(col_widths[col]) for col in columns)
        print(f"\n{header}")
        print("-" * len(header))

        # Print rows
        for row in results:
            row_str = []
            for col in columns:
                value = row.get(col, '')
                if col == 'embedding' and value:
                    if isinstance(value, list):
                        value = f"[{len(value)}D vector]"
                    else:
                        value = str(value)[:20] + "..."
                else:
                    value = str(value)

                row_str.append(value[:col_widths[col]].ljust(col_widths[col]))

            print(" | ".join(row_str))

    def _print_csv(self, results):
        """Print results in CSV format."""
        if not results:
            return

        import csv
        import io

        output = io.StringIO()
        columns = list(results[0].keys())
        writer = csv.DictWriter(output, fieldnames=columns)

        writer.writeheader()
        for row in results:
            # Convert embedding to string representation
            csv_row = row.copy()
            if 'embedding' in csv_row and csv_row['embedding']:
                if isinstance(csv_row['embedding'], list):
                    csv_row['embedding'] = f"[{len(csv_row['embedding'])}D vector]"
            writer.writerow(csv_row)

        print(output.getvalue())

    def _save_to_file(self, results, filename, format_type):
        """Save results to file."""
        try:
            if format_type == "json":
                with open(filename, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
            elif format_type == "csv":
                import csv
                with open(filename, 'w', newline='') as f:
                    if results:
                        columns = list(results[0].keys())
                        writer = csv.DictWriter(f, fieldnames=columns)
                        writer.writeheader()
                        for row in results:
                            csv_row = row.copy()
                            if 'embedding' in csv_row and csv_row['embedding']:
                                if isinstance(csv_row['embedding'], list):
                                    csv_row['embedding'] = f"[{len(csv_row['embedding'])}D vector]"
                            writer.writerow(csv_row)

            print(f"💾 Results saved to: {filename}")
        except Exception as e:
            print(f"❌ Error saving to file: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Retrieve embeddings from pgVector PostgreSQL database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all databases
  python retrieve_embeddings.py --list-databases

  # List tables in a specific database
  python retrieve_embeddings.py --list-tables --database "text_and_images_ejeannot"

  # Retrieve all embeddings for user ejeannot from text table
  python retrieve_embeddings.py --user ejeannot --table text --limit 10

  # Retrieve embeddings with date filter
  python retrieve_embeddings.py --user ejeannot --experiment "text_and_images" \\
                               --date-from 2025-01-01 --date-to 2025-07-10

  # Search for similar embeddings
  python retrieve_embeddings.py --user ejeannot --table text \\
                               --similarity "apple fruit delicious" --top-k 5

  # Export results to JSON
  python retrieve_embeddings.py --user ejeannot --table image \\
                               --output-format json --save-to embeddings.json

  # Filter by S3 object name
  python retrieve_embeddings.py --user ejeannot --table text \\
                               --s3-filter "Cat_" --limit 20
        """
    )

    # Connection parameters
    parser.add_argument("--host", default="localhost", help="PostgreSQL host")
    parser.add_argument("--port", default="5432", help="PostgreSQL port")
    parser.add_argument("--superuser", default="postgres", help="PostgreSQL superuser")
    parser.add_argument("--password", default="postgres", help="PostgreSQL password")

    # Database and table selection
    parser.add_argument("--user", help="Database username")
    parser.add_argument("--experiment", default="text_and_images",
                       help="Experiment name (default: text_and_images)")
    parser.add_argument("--database", help="Specific database name (overrides user+experiment)")
    parser.add_argument("--table", help="Table name")

    # Listing options
    parser.add_argument("--list-databases", action="store_true",
                       help="List all available databases")
    parser.add_argument("--list-tables", action="store_true",
                       help="List tables with vector columns")

    # Filtering options
    parser.add_argument("--limit", type=int, help="Maximum number of records to retrieve")
    parser.add_argument("--date-from", help="Start date filter (YYYY-MM-DD)")
    parser.add_argument("--date-to", help="End date filter (YYYY-MM-DD)")
    parser.add_argument("--s3-filter", help="Filter by S3 object name (supports wildcards)")

    # Similarity search
    parser.add_argument("--similarity", help="Text query for similarity search")
    parser.add_argument("--similarity-model", default="all-MiniLM-L6-v2",
                       help="Model for similarity search (default: all-MiniLM-L6-v2)")
    parser.add_argument("--top-k", type=int, default=10,
                       help="Number of similar results (default: 10)")

    # Output options
    parser.add_argument("--output-format", choices=["table", "json", "csv"],
                       default="table", help="Output format")
    parser.add_argument("--save-to", help="Save results to file")

    args = parser.parse_args()

    # Create retriever instance
    retriever = EmbeddingsRetriever(args.host, args.port, args.superuser, args.password)

    try:
        # Handle listing operations
        if args.list_databases:
            retriever.list_databases(args.user)
            return

        if args.list_tables:
            if args.database:
                retriever.list_tables(args.database)
            elif args.user:
                db_name = retriever.build_database_name(args.user, args.experiment)
                retriever.list_tables(db_name)
            else:
                print("❌ Please specify --database or --user for listing tables")
            return

        # Validate required parameters for retrieval
        if not args.user:
            print("❌ --user is required for retrieving embeddings")
            return

        # Retrieve embeddings
        results = retriever.retrieve_embeddings(
            username=args.user,
            experiment=args.experiment,
            table=args.table,
            limit=args.limit,
            date_from=args.date_from,
            date_to=args.date_to,
            s3_object_filter=args.s3_filter,
            similarity_query=args.similarity,
            similarity_model=args.similarity_model,
            top_k=args.top_k,
            output_format=args.output_format,
            save_to_file=args.save_to
        )

        if results:
            print(f"\n📊 Summary: Retrieved {len(results)} embedding records")

    finally:
        retriever.disconnect()


if __name__ == "__main__":
    main()
