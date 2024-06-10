# Databricks notebook source
df = spark.table("bitext.default.bitext_customer_service_training_delta")
display(df.where("intent != 'cancel_order'"))

# COMMAND ----------

import pyspark.sql.functions as F

df.withColumn("id", F.monotonically_increasing_id()).withColumn(
    "intent_and_response",
    F.concat(
        F.col("instruction"),
        F.lit(" - "),
        F.col("intent"),
        F.lit(" - "),
        F.col("response"),
    ),
).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    "workspace.default.customer_service_data3"
)

# COMMAND ----------

display(spark.table("workspace.default.customer_service_data3"))
