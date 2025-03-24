
-- Create Utilization Results Table (U1)
CREATE TABLE U1 (
    asecid TEXT,
    date TEXT,
    cusip TEXT,
    isin TEXT,
    tradingsymbol TEXT,
    utilizationpercentunits REAL
);

-- Create Rates Table (R1)
CREATE TABLE R1 (
    tradingsymbol TEXT
);

-- Join query used in analysis (based on available columns)
SELECT 
    u.tradingsymbol,
    u.date,
    u.utilizationpercentunits,
    r.tradingsymbol AS r_symbol
FROM 
    U1 u
JOIN 
    R1 r 
ON 
    u.tradingsymbol = r.tradingsymbol;
