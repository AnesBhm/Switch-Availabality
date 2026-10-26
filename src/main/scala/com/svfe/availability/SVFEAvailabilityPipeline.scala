package com.svfe.availability

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.types.{IntegerType, LongType, TimestampType}

/**
 * ═══════════════════════════════════════════════════════════════════════════════
 *  SVFE Switch Availability Pipeline — UseCase 06
 * ═══════════════════════════════════════════════════════════════════════════════
 *
 *  PURPOSE
 *  -------
 *  Calculate the daily availability (Uptime vs. Downtime) of the SmartVista
 *  Front End (SVFE) financial payment switch.
 *
 *  This pipeline leverages explicit response code mapping to ensure ONLY true
 *  internal SVFE crashes are recorded as downtime, filtering out normal business
 *  declines (e.g. Insufficient Funds) and issuer unavailability.
 *
 *  SLA PASS condition: Availability >= 99.99 % (max downtime <= 8.64 s/day)
 * ═══════════════════════════════════════════════════════════════════════════════
 */
object SVFEAvailabilityPipeline {

  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder()
      .appName("SVFE Availability Pipeline — UseCase 06")
      .master("local[*]")
      .config("spark.sql.session.timeZone", "UTC")
      .config("spark.sql.legacy.timeParserPolicy", "LEGACY")
      .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")
    import spark.implicits._

    println("=" * 80)
    println("  SVFE AVAILABILITY PIPELINE — UseCase 06")
    println("=" * 80)

    // ═════════════════════════════════════════════════════════════════════════
    //  STEP 1 — DATABASE INGESTION
    // ═════════════════════════════════════════════════════════════════════════
    println("\n[STEP 1] Ingesting raw posting file data from PostgreSQL...")

    val jdbcUrl  = "jdbc:postgresql://dsip_shared_postgres_db:5432/UseCase04"
    val jdbcUser = "postgres"
    val jdbcPass = "postgres"
    val table    = "ops_20231101_012920_allnw"

    val rawDF = spark.read
      .format("jdbc")
      .option("url",      jdbcUrl)
      .option("dbtable",  table)
      .option("user",     jdbcUser)
      .option("password", jdbcPass)
      .option("driver",   "org.postgresql.Driver")
      .option("fetchsize", "10000")
      .load()

    val rawCount = rawDF.count()
    println(s"   ✓ Ingested $rawCount raw rows from table '$table'.")

    // ═════════════════════════════════════════════════════════════════════════
    //  STEP 2 — STRING PARSING & DATA CLEANING
    // ═════════════════════════════════════════════════════════════════════════
    println("\n[STEP 2] Parsing fixed-width fields & cleaning data...")

    val recordType = trim(substring(col("line"), 1, 1))

    val dataRowsDF = rawDF
      .withColumn("record_type", recordType)
      .filter(col("record_type") =!= "1" && col("record_type") =!= "@")

    val filteredCount = dataRowsDF.count()

    val parsedDF = dataRowsDF
      // ── Primary Keys & Identification ──
      .withColumn("pan",              trim(substring(col("line"),   1, 19)))
      .withColumn("system_trace",     trim(substring(col("line"),  96,  6)))
      .withColumn("svfe_trace",       trim(substring(col("line"), 112,  9)))

      // ── Timeline Core ──
      .withColumn("svfe_time_raw",    trim(substring(col("line"), 126,  6)))  // hhmmss
      .withColumn("svfe_date_raw",    trim(substring(col("line"), 132,  4)))  // MMDD

      // ── Diagnostic Core ──
      .withColumn("network_resp_code",  trim(substring(col("line"), 312, 2)))  // F44
      .withColumn("svfe_resp_code",     trim(substring(col("line"), 780, 3)))  // F66
      .withColumn("reversal_flag",      trim(substring(col("line"), 111, 1)).cast(IntegerType))  // F12
      .withColumn("completion_status",  trim(substring(col("line"), 879, 1)).cast(IntegerType))  // F76
      .withColumn("stood_in_for",       trim(substring(col("line"), 880, 1)).cast(IntegerType))  // F77
      .withColumn("issuer_posted",      trim(substring(col("line"), 881, 1)).cast(IntegerType))  // F78

      // ── Scoping & Routing ──
      .withColumn("acquiring_inst",   trim(substring(col("line"), 244, 11)))  // F39
      .withColumn("terminal_type",    trim(substring(col("line"), 874,  5)))  // F75
      .withColumn("issuing_inst",     trim(substring(col("line"), 890, 11)))  // F81
      .drop("line", "record_type")

    val withTimestampDF = parsedDF
      .withColumn(
        "server_timestamp_str",
        concat(lit("2023"), col("svfe_date_raw"), col("svfe_time_raw"))
      )
      .withColumn(
        "server_timestamp",
        to_timestamp(col("server_timestamp_str"), "yyyyMMddHHmmss")
      )

    val validTimestampDF = withTimestampDF.filter(col("server_timestamp").isNotNull)
    val dedupedDF = validTimestampDF.dropDuplicates(Seq("pan", "system_trace", "svfe_trace", "server_timestamp"))

    val dedupCount  = dedupedDF.count()
    println(s"   ✓ Deduplication complete: $dedupCount unique valid transactions.")

    // ═════════════════════════════════════════════════════════════════════════
    //  STEP 3 — TIME-SERIES FEATURE ENGINEERING (Δt)
    // ═════════════════════════════════════════════════════════════════════════
    println("\n[STEP 3] Computing inter-transaction time gaps (Δt)...")

    val globalWindow = Window.orderBy("server_timestamp")

    val withDeltaDF = dedupedDF
      .withColumn("prev_timestamp", lag("server_timestamp", 1).over(globalWindow))
      .withColumn(
        "delta_t_seconds",
        when(col("prev_timestamp").isNotNull,
          unix_timestamp(col("server_timestamp")) - unix_timestamp(col("prev_timestamp"))
        ).otherwise(lit(0L))
      )

    println(s"   ✓ Δt (delta_t_seconds) computed.")

    // ═════════════════════════════════════════════════════════════════════════
    //  STEP 4 — DATA-DRIVEN DIAGNOSTIC LOGIC (Fault Isolation)
    // ═════════════════════════════════════════════════════════════════════════
    println("\n[STEP 4] Applying fault-isolation rules for SVFE downtime...")

    // Specific SVFE internal malfunction codes mapped from the CSV reference.
    // -01 is OK, anything else is typically a business decline unless explicitly defined here.
    val internalSvfeErrors = Seq("000", "951", "959", "974", "975", "805")

    val diagnosticDF = withDeltaDF
      .withColumn(
        "svfe_downtime_seconds",
        when(
          // ── RULE 1: Massive Gap / Total Crash ──
          col("delta_t_seconds") > 300,
          col("delta_t_seconds")
        )
        .when(
          // ── RULE 4 (Exclusion): Stand-In Success ──
          // SVFE successfully stood in for a dead issuer. SVFE is UP.
          (col("stood_in_for") === 1) && (col("issuer_posted") === 0),
          lit(0L)
        )
        .when(
          // ── RULE 2: SVFE Internal Failure on a Good Network Response ──
          // Network approved ("00"), but SVFE didn't record OK ("-01").
          (col("network_resp_code") === "00") && (col("svfe_resp_code") =!= "-01"),
          col("delta_t_seconds")
        )
        .when(
          // ── RULE 3: Explicit SVFE System Malfunction ──
          // Network threw an error, AND SVFE threw an explicit internal failure code.
          // This properly filters out standard business declines (like Insufficient Funds).
          (col("network_resp_code") =!= "00") && col("svfe_resp_code").isin(internalSvfeErrors: _*),
          col("delta_t_seconds")
        )
        .when(
          // ── RULE 5: Suspect Timeout ──
          // Completion status 0 means SVFE/HSM/DB timed out locally
          col("completion_status") === 0 && col("svfe_resp_code") =!= "-01",
          col("delta_t_seconds")
        )
        .otherwise(
          // ── DEFAULT: Normal Operations or External Business Declines ──
          lit(0L)
        )
      )

    // ═════════════════════════════════════════════════════════════════════════
    //  STEP 5 — KPI AGGREGATION & SLA RATE CALCULATION
    // ═════════════════════════════════════════════════════════════════════════
    println("\n[STEP 5] Aggregating KPIs and computing SLA availability rate...")

    val totalSecondsInDay = 86400.0

    val kpiRow = diagnosticDF
      .agg(
        sum("svfe_downtime_seconds").alias("total_downtime_seconds"),
        count("*").alias("total_transactions"),
        sum(when(col("svfe_downtime_seconds") > 0, 1).otherwise(0)).alias("downtime_events"),
        max("svfe_downtime_seconds").alias("max_single_gap_seconds")
      )
      .head()

    val totalDowntimeSeconds   = Option(kpiRow.getAs[Long]("total_downtime_seconds")).getOrElse(0L)
    val totalTransactions      = kpiRow.getAs[Long]("total_transactions")
    val downtimeEvents         = Option(kpiRow.getAs[Long]("downtime_events")).getOrElse(0L)
    val maxSingleGapSeconds    = Option(kpiRow.getAs[Long]("max_single_gap_seconds")).getOrElse(0L)
    val totalDowntimeMinutes   = totalDowntimeSeconds / 60.0
    val availabilityPercentage = ((totalSecondsInDay - totalDowntimeSeconds) / totalSecondsInDay) * 100.0
    val slaTarget              = 99.99
    val meetsSlA               = availabilityPercentage >= slaTarget

    println("\n   ── Downtime Breakdown by Trigger Rule ──")
    diagnosticDF
      .withColumn(
        "trigger_rule",
        when(col("svfe_downtime_seconds") === 0, "NORMAL_OR_EXTERNAL_BUSINESS_DECLINE")
        .when(col("delta_t_seconds") > 300, "RULE_1_MASSIVE_GAP")
        .when(
          (col("network_resp_code") === "00") && (col("svfe_resp_code") =!= "-01"),
          "RULE_2_SVFE_FAILED_GOOD_NETWORK"
        )
        .when(
          (col("network_resp_code") =!= "00") && col("svfe_resp_code").isin(internalSvfeErrors: _*),
          "RULE_3_EXPLICIT_SVFE_MALFUNCTION"
        )
        .when(
          col("completion_status") === 0 && col("svfe_resp_code") =!= "-01",
          "RULE_5_SUSPECT_TIMEOUT"
        )
        .otherwise("UNKNOWN")
      )
      .groupBy("trigger_rule")
      .agg(
        count("*").alias("event_count"),
        sum("svfe_downtime_seconds").alias("total_seconds")
      )
      .orderBy(col("total_seconds").desc)
      .show(truncate = false)

    // ═════════════════════════════════════════════════════════════════════════
    //  FINAL REPORT
    // ═════════════════════════════════════════════════════════════════════════
    println("=" * 80)
    println("  SVFE AVAILABILITY REPORT — UseCase 06")
    println("=" * 80)
    println(f"  Total Transactions     : $totalTransactions%,d")
    println(f"  Downtime Events        : $downtimeEvents%,d")
    println(f"  Max Single Gap         : $maxSingleGapSeconds%,d seconds")
    println("-" * 80)
    println(f"  Total Downtime         : $totalDowntimeSeconds%,d seconds  ($totalDowntimeMinutes%.2f minutes)")
    println(f"  Availability Rate      : $availabilityPercentage%.6f %%")
    println(f"  SLA Target             : $slaTarget%.2f %%")
    println("-" * 80)

    if (meetsSlA) {
      println(s"  ✅ SLA STATUS: PASS — System meets the $slaTarget% availability standard.")
    } else {
      val breachSeconds = totalDowntimeSeconds - (totalSecondsInDay * (1.0 - slaTarget / 100.0))
      println(f"  ❌ SLA STATUS: FAIL — System does NOT meet the $slaTarget%.2f%% availability standard.")
      println(f"     Downtime exceeded SLA threshold by $breachSeconds%.2f seconds.")
    }
    println("=" * 80)

    spark.stop()
  }
}
