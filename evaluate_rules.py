import json
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType

def evaluate_data_quality_rules(spark: SparkSession, df, rules_json_path: str, table_name: str = "withdeltadf"):
    """
    Evaluates data quality rules from a JSON file against a PySpark DataFrame.
    
    Args:
        spark: The active SparkSession
        df: The DataFrame to evaluate
        rules_json_path: Path to the JSON file containing the rules
        table_name: The table name to use for the temp view (default 'withdeltadf')
        
    Returns:
        A PySpark DataFrame containing the evaluation results.
    """
    
    # Load rules from the JSON file
    with open(rules_json_path, 'r') as f:
        data = json.load(f)
        
    rules = data.get("rules", [])
    
    # Register the dataframe as a temporary view so SQL queries can reference it
    df.createOrReplaceTempView(table_name)
    
    total_df_rows = df.count()
    results = []
    
    for rule in rules:
        rule_sql = rule.get("ruleSql")
        col_name = rule.get("columnName")
        dimension = rule.get("dimension")
        severity = rule.get("level")
        rule_table_name = rule.get("table", table_name)
        
        try:
            # Execute the rule SQL
            res_df = spark.sql(rule_sql)
            columns = res_df.columns
            
            failed_rows = 0
            total_rows = total_df_rows
            
            # --- Heuristics to analyze rule results based on rule types ---
            
            # Pattern 1: Aggregation queries returning total and specific metric counts
            if "total_records" in columns:
                row = res_df.collect()[0]
                total_rows = row["total_records"]
                if "invalid_records" in columns:
                    failed_rows = row["invalid_records"] or 0
                elif "consistent_records" in columns:
                    failed_rows = total_rows - (row["consistent_records"] or 0)
                elif "unique_records" in columns:
                    failed_rows = total_rows - (row["unique_records"] or 0)
                    
            # Pattern 2: Queries that return a 'record_status' classification column
            elif "record_status" in columns:
                failed_rows = res_df.filter(col("record_status") != "valid_record").count()
                total_rows = total_df_rows
                
            # Pattern 3: Standard SELECT * FROM ... WHERE ...
            else:
                res_count = res_df.count()
                total_rows = total_df_rows
                
                sql_upper = rule_sql.upper()
                
                # If the query is specifically looking for invalid data
                if " NOT IN " in sql_upper or " NOT (" in sql_upper or " NOT(" in sql_upper or "> CURRENT_TIMESTAMP" in sql_upper:
                    failed_rows = res_count
                else:
                    # If the query is validating data (returns the valid records)
                    failed_rows = total_rows - res_count
                    
            # Calculate metrics
            failed_rows = int(failed_rows)
            total_rows = int(total_rows)
            
            if total_rows == 0:
                quality_score = 100.0
                comparison_score = "0/0"
            else:
                quality_score = max(0.0, 100.0 - (failed_rows / total_rows * 100.0))
                comparison_score = f"{failed_rows}/{total_rows}"
                
            rule_status = "PASS" if failed_rows == 0 else "FAIL"
            
            results.append({
                "ruleSql": rule_sql,
                "tableName": rule_table_name,
                "columnName": col_name,
                "dimension": dimension,
                "severity": severity,
                "ruleStatus": rule_status,
                "failedRows": failed_rows,
                "totalRows": total_rows,
                "comparisonScore": comparison_score,
                "qualityScore": round(quality_score, 2)
            })
            
        except Exception as e:
            # Handle possible SQL execution errors (e.g. malformed queries)
            print(f"Error evaluating rule {rule.get('ruleId', 'Unknown')}: {str(e)}")
            results.append({
                "ruleSql": rule_sql,
                "tableName": rule_table_name,
                "columnName": col_name,
                "dimension": dimension,
                "severity": severity,
                "ruleStatus": "ERROR",
                "failedRows": -1,
                "totalRows": total_df_rows,
                "comparisonScore": "ERROR",
                "qualityScore": 0.0
            })

    # Define exact schema mapping to the desired output
    schema = StructType([
        StructField("ruleSql", StringType(), True),
        StructField("tableName", StringType(), True),
        StructField("columnName", StringType(), True),
        StructField("dimension", StringType(), True),
        StructField("severity", StringType(), True),
        StructField("ruleStatus", StringType(), True),
        StructField("failedRows", IntegerType(), True),
        StructField("totalRows", IntegerType(), True),
        StructField("comparisonScore", StringType(), True),
        StructField("qualityScore", DoubleType(), True)
    ])
    
    results_df = spark.createDataFrame(results, schema=schema)
    return results_df

if __name__ == "__main__":
    # Example usage block
    # Initialize Spark Session
    spark = SparkSession.builder.appName("DataQualityEvaluator").getOrCreate()
    
    # 1. Load your actual data frame here
    # df = spark.read.parquet("path_to_your_data")
    
    # 2. Path to your JSON rules file
    # rules_path = r"C:\Users\B-TECH\Downloads\SWITCH AVAILIBILITY_rules_2026-07-20.json"
    
    # 3. Call the function
    # results_df = evaluate_data_quality_rules(spark, df, rules_path, "withdeltadf")
    
    # 4. Display the results
    # results_df.show(truncate=False)
