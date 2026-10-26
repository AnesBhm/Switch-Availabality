import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._
import scala.util.Try

// 1. You already have your dataframe loaded
// val df = cleanedDF123
// Replace this with your actual dataframe variable
val df: DataFrame = cleanedDF123
df.createOrReplaceTempView("withdeltadf")

val totalDfRows = df.count()

// 2. Since the JSON is locally saved and you are on a remote platform,
// the easiest way is to paste the JSON content directly as a multiline string.
val rulesJsonString = """
// PASTE THE ENTIRE CONTENT OF YOUR JSON FILE HERE
// Example:
// {
//   "projectName": "SWITCH AVAILIBILITY",
//   "rules": [
//     {
//       "ruleId": "354",
//       "ruleSql": "SELECT * FROM withdeltadf WHERE svfe_resp_code IS NOT  NULL;",
//       "table": "withdeltadf",
//       "columnName": "svfe_resp_code",
//       "dimension": "Completeness",
//       "level": "critical"
//     }
//   ]
// }
"""

// 3. Read the JSON string into a Spark DataFrame to extract the rules
import spark.implicits._
val rulesDF = spark.read.json(Seq(rulesJsonString).toDS())
val rulesArray = rulesDF.select(explode(col("rules")).as("rule")).select("rule.*").collect()

// Helper to safely extract Longs regardless of exact numeric type (Int/Long)
def getAsLong(r: Row, field: String): Long = {
    Try(r.getAs[Any](field) match {
        case n: Number => n.longValue()
        case s: String => s.toLong
        case _ => 0L
    }).getOrElse(0L)
}

// Helper to safely extract Strings
def getAsString(r: Row, field: String): String = {
    Try(r.getAs[Any](field) match {
        case null => null
        case any => any.toString
    }).getOrElse(null)
}

var resultsSeq = Seq[Row]()

for (row <- rulesArray) {
    val ruleSql = getAsString(row, "ruleSql")
    val colName = getAsString(row, "columnName")
    val dimension = getAsString(row, "dimension")
    val severity = getAsString(row, "level")
    val tableName = Try(getAsString(row, "table")).filter(_ != null).getOrElse("withdeltadf")
    
    var failedRows = 0L
    var totalRows = totalDfRows
    
    try {
        // Execute the SQL for the rule
        val resDf = spark.sql(ruleSql)
        val cols = resDf.columns
        
        // --- Heuristics to analyze rule results based on query pattern ---
        if (cols.contains("total_records")) {
            val resRow = resDf.collect().head
            totalRows = getAsLong(resRow, "total_records")
            
            if (cols.contains("invalid_records")) {
                failedRows = getAsLong(resRow, "invalid_records")
            } else if (cols.contains("consistent_records")) {
                failedRows = totalRows - getAsLong(resRow, "consistent_records")
            } else if (cols.contains("unique_records")) {
                failedRows = totalRows - getAsLong(resRow, "unique_records")
            }
        } else if (cols.contains("record_status")) {
            failedRows = resDf.filter(col("record_status") =!= "valid_record").count()
        } else {
            val resCount = resDf.count()
            val sqlUpper = ruleSql.toUpperCase()
            
            if (sqlUpper.contains(" NOT IN ") || sqlUpper.contains(" NOT (") || sqlUpper.contains(" NOT(") || sqlUpper.contains("> CURRENT_TIMESTAMP")) {
                failedRows = resCount
            } else {
                failedRows = totalRows - resCount
            }
        }
        
        val qualityScore: Double = if (totalRows == 0) 100.0 else math.max(0.0, 100.0 - (failedRows.toDouble / totalRows * 100.0))
        val ruleStatus = if (failedRows == 0) "PASS" else "FAIL"
        val comparisonScore = s"$failedRows/$totalRows"
        
        val roundedQualityScore = BigDecimal(qualityScore).setScale(2, BigDecimal.RoundingMode.HALF_UP).toDouble
        
        resultsSeq = resultsSeq :+ Row(ruleSql, tableName, colName, dimension, severity, ruleStatus, failedRows, totalRows, comparisonScore, roundedQualityScore)
        
    } catch {
        case e: Exception => 
            println(s"Error evaluating rule: ${e.getMessage}")
            resultsSeq = resultsSeq :+ Row(ruleSql, tableName, colName, dimension, severity, "ERROR", -1L, totalRows, "ERROR", 0.0)
    }
}

// 4. Create the final results DataFrame
val schema = StructType(Array(
    StructField("ruleSql", StringType, true),
    StructField("tableName", StringType, true),
    StructField("columnName", StringType, true),
    StructField("dimension", StringType, true),
    StructField("severity", StringType, true),
    StructField("ruleStatus", StringType, true),
    StructField("failedRows", LongType, true),
    StructField("totalRows", LongType, true),
    StructField("comparisonScore", StringType, true),
    StructField("qualityScore", DoubleType, true)
))

val resultsDF = spark.createDataFrame(spark.sparkContext.parallelize(resultsSeq), schema)

// Display the results exactly like your screenshot
resultsDF.show(truncate = false)
