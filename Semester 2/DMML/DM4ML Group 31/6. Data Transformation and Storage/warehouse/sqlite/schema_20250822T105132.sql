-- Auto-extracted schema for SQLite tables

CREATE TABLE generic_features (
	row_id BIGINT, 
	gender BIGINT, 
	"SeniorCitizen" FLOAT, 
	"Partner" BIGINT, 
	"Dependents" BIGINT, 
	tenure FLOAT, 
	"PhoneService" BIGINT, 
	"Churn" BIGINT, 
	"MultipleLines_No phone service" BOOLEAN, 
	"MultipleLines_Yes" BOOLEAN, 
	"InternetService_Fiber optic" BOOLEAN, 
	"InternetService_No" BOOLEAN, 
	"OnlineSecurity_No internet service" BOOLEAN, 
	"OnlineSecurity_Yes" BOOLEAN, 
	"OnlineBackup_No internet service" BOOLEAN, 
	"OnlineBackup_Yes" BOOLEAN, 
	services_count BIGINT, 
	services_count_minmax FLOAT, 
	services_count_z FLOAT
);
