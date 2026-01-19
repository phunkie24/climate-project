"""
Data validation and quality checks
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


class DataValidator:
    """Validate and check data quality"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.quality_report = {}
    
    def check_completeness(self) -> Dict:
        """Check for missing values"""
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100
        
        self.quality_report['completeness'] = {
            'missing_counts': missing.to_dict(),
            'missing_percentages': missing_pct.to_dict(),
            'total_rows': len(self.df),
            'complete_rows': len(self.df.dropna())
        }
        
        return self.quality_report['completeness']
    
    def check_validity(self) -> Dict:
        """Check if values are within expected ranges"""
        rules = {
            'precipitation(mm)': (0, 10000),
            'avg_temp_c': (-50, 60),
            'avg_humidity(%)': (0, 100),
            'cloud_cover(%)': (0, 100),
            'atmospheric_co2(ppm)': (300, 500)
        }
        
        violations = {}
        for col, (min_val, max_val) in rules.items():
            if col in self.df.columns:
                invalid = self.df[
                    (self.df[col] < min_val) | 
                    (self.df[col] > max_val)
                ]
                if len(invalid) > 0:
                    violations[col] = {
                        'count': len(invalid),
                        'percentage': (len(invalid) / len(self.df)) * 100
                    }
        
        self.quality_report['validity'] = violations
        return violations
    
    def check_consistency(self) -> Dict:
        """Check temporal and spatial consistency"""
        issues = {}
        
        # Check for duplicate country-year combinations
        if 'country' in self.df.columns and 'year' in self.df.columns:
            duplicates = self.df.duplicated(subset=['country', 'year'])
            issues['duplicate_records'] = duplicates.sum()
        
        # Check year sequence
        if 'year' in self.df.columns:
            years = sorted(self.df['year'].unique())
            expected_years = list(range(min(years), max(years) + 1))
            missing_years = set(expected_years) - set(years)
            issues['missing_years'] = list(missing_years)
        
        self.quality_report['consistency'] = issues
        return issues
    
    def check_outliers(self, threshold: float = 4.0) -> Dict:
        """Detect statistical outliers using z-score"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        outliers = {}
        for col in numeric_cols:
            if self.df[col].std() > 0:  # Avoid division by zero
                z_scores = np.abs((self.df[col] - self.df[col].mean()) / self.df[col].std())
                extreme_outliers = (z_scores > threshold).sum()
                if extreme_outliers > 0:
                    outliers[col] = {
                        'count': extreme_outliers,
                        'percentage': (extreme_outliers / len(self.df)) * 100
                    }
        
        self.quality_report['outliers'] = outliers
        return outliers
    
    def generate_report(self) -> Dict:
        """Generate comprehensive quality report"""
        print("="*60)
        print("DATA QUALITY REPORT")
        print("="*60)
        
        # Completeness
        completeness = self.check_completeness()
        print("\n[1] COMPLETENESS")
        print(f"Total rows: {completeness['total_rows']}")
        print(f"Complete rows: {completeness['complete_rows']}")
        
        missing_cols = {k: v for k, v in completeness['missing_percentages'].items() if v > 0}
        if missing_cols:
            print("\nMissing values:")
            for col, pct in sorted(missing_cols.items(), key=lambda x: x[1], reverse=True):
                print(f"  {col}: {pct:.1f}%")
        else:
            print("[OK] No missing values")
        
        # Validity
        validity = self.check_validity()
        print("\n[2] VALIDITY")
        if validity:
            print("Range violations detected:")
            for col, info in validity.items():
                print(f"  {col}: {info['count']} records ({info['percentage']:.1f}%)")
        else:
            print("[OK] All values within expected ranges")
        
        # Consistency
        consistency = self.check_consistency()
        print("\n[3] CONSISTENCY")
        if consistency.get('duplicate_records', 0) > 0:
            print(f"[WARN] Duplicate records: {consistency['duplicate_records']}")
        else:
            print("[OK] No duplicate records")
        
        if consistency.get('missing_years'):
            print(f"[WARN] Missing years: {consistency['missing_years']}")
        else:
            print("[OK] Complete year sequence")
        
        # Outliers
        outliers = self.check_outliers()
        print("\n[4] OUTLIERS (>4 std dev)")
        if outliers:
            for col, info in outliers.items():
                print(f"  {col}: {info['count']} outliers ({info['percentage']:.1f}%)")
        else:
            print("[OK] No extreme outliers detected")
        
        print("\n" + "="*60)
        
        return self.quality_report


def validate_dataframe(df: pd.DataFrame, verbose: bool = True) -> Tuple[bool, Dict]:
    """
    Validate a DataFrame
    
    Args:
        df: DataFrame to validate
        verbose: Print detailed report
        
    Returns:
        Tuple of (is_valid, report)
    """
    validator = DataValidator(df)
    report = validator.generate_report() if verbose else validator.quality_report
    
    # Determine if data is valid
    is_valid = (
        report.get('validity', {}) == {} and
        report.get('consistency', {}).get('duplicate_records', 0) == 0
    )
    
    return is_valid, report
