"""
O2TAB-GAN: Orthopaedic Oncology Tabular GAN
Author: Dr. Ehsan Pendar
Date: July 4, 2025
Description: Initial data analysis and preprocessing for SEER orthopaedic-oncology dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sdv.metadata import SingleTableMetadata
from typing import Union, Optional
import warnings
warnings.filterwarnings('ignore')

def load_and_inspect_data(data_path: str) -> pd.DataFrame:
    """
    Load the SEER dataset and perform initial inspection.
    
    Args:
        data_path: Path to the CSV file
        
    Returns:
        pandas DataFrame with the loaded data
    """
    try:
        print("Loading SEER orthopaedic-oncology dataset...")
        data = pd.read_csv(data_path)
        print(f"✅ Dataset loaded successfully!")
        print(f"📊 Shape: {data.shape}")
        return data
    except FileNotFoundError:
        print(f"❌ Error: File not found at {data_path}")
        return None
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

def basic_data_inspection(data: pd.DataFrame) -> None:
    """
    Perform basic data inspection and display key statistics.
    
    Args:
        data: pandas DataFrame to inspect
    """
    print("\n" + "="*60)
    print("BASIC DATA INSPECTION")
    print("="*60)
    
    print(f"\n📋 Dataset Info:")
    print(f"   - Rows: {data.shape[0]:,}")
    print(f"   - Columns: {data.shape[1]}")
    print(f"   - Memory usage: {data.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    print(f"\n📊 Data Types:")
    data_types = data.dtypes.value_counts()
    for dtype, count in data_types.items():
        print(f"   - {dtype}: {count} columns")
    
    print(f"\n🔍 Missing Values:")
    missing_data = data.isnull().sum()
    if missing_data.sum() > 0:
        print(f"   - Total missing values: {missing_data.sum():,}")
        print(f"   - Columns with missing values: {(missing_data > 0).sum()}")
        for col, missing_count in missing_data[missing_data > 0].items():
            print(f"     * {col}: {missing_count:,} ({missing_count/len(data)*100:.1f}%)")
    else:
        print("   - No missing values found ✅")
    
    print(f"\n📈 Numerical Columns Summary:")
    numerical_cols = data.select_dtypes(include=[np.number]).columns
    if len(numerical_cols) > 0:
        print(data[numerical_cols].describe())
    else:
        print("   - No numerical columns found")
    
    print(f"\n🏷️ Categorical Columns Cardinality:")
    categorical_cols = data.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        cardinality = data[categorical_cols].nunique().sort_values(ascending=False)
        for col, unique_count in cardinality.items():
            print(f"   - {col}: {unique_count:,} unique values")
    else:
        print("   - No categorical columns found")

def analyze_high_cardinality_columns(data: pd.DataFrame) -> None:
    """
    Analyze high-cardinality categorical columns in detail.
    
    Args:
        data: pandas DataFrame to analyze
    """
    print("\n" + "="*60)
    print("HIGH-CARDINALITY ANALYSIS")
    print("="*60)
    
    categorical_cols = data.select_dtypes(include=['object']).columns
    high_cardinality_cols = []
    
    for col in categorical_cols:
        unique_count = data[col].nunique()
        if unique_count > 10:  # Threshold for high cardinality
            high_cardinality_cols.append((col, unique_count))
    
    if high_cardinality_cols:
        print(f"\n🔍 High-cardinality columns (>10 unique values):")
        for col, count in sorted(high_cardinality_cols, key=lambda x: x[1], reverse=True):
            print(f"\n   📊 {col}: {count:,} unique values")
            print(f"      Top 10 most frequent values:")
            top_values = data[col].value_counts().head(10)
            for value, freq in top_values.items():
                print(f"        - {value}: {freq:,} ({freq/len(data)*100:.1f}%)")
    else:
        print("\n   No high-cardinality columns found")

def create_metadata_schema(data: pd.DataFrame) -> SingleTableMetadata:
    """
    Create and validate SDV metadata schema for the dataset.
    
    Args:
        data: pandas DataFrame
        
    Returns:
        SDV SingleTableMetadata object
    """
    print("\n" + "="*60)
    print("METADATA SCHEMA CREATION")
    print("="*60)
    
    # Initialize metadata
    metadata = SingleTableMetadata()
    
    # Auto-detect schema
    print("\n🔍 Auto-detecting metadata schema...")
    metadata.detect_from_dataframe(data=data)
    
    # Manual adjustments based on domain knowledge
    print("\n🔧 Applying domain-specific adjustments...")
    
    # Ensure numerical columns are properly typed
    numerical_candidates = ['Year of diagnosis', 'Age recode with <1 year olds', 'Survival months']
    for col in numerical_candidates:
        if col in data.columns:
            metadata.update_column(column_name=col, sdtype='numerical')
            print(f"   ✅ Set '{col}' as numerical")
    
    # Validate metadata
    print("\n✅ Validating metadata schema...")
    try:
        metadata.validate()
        print("   Metadata validation successful!")
    except Exception as e:
        print(f"   ❌ Metadata validation failed: {e}")
        return None
    
    # Save metadata
    metadata_path = 'seer_metadata.json'
    metadata.save_to_json(metadata_path)
    print(f"   💾 Metadata saved to: {metadata_path}")
    
    return metadata

def prepare_conditional_vectors(data: pd.DataFrame, conditional_column: Optional[str] = None) -> Optional[dict]:
    """
    Prepare conditional vectors for PacGAN training.
    
    Args:
        data: pandas DataFrame
        conditional_column: Column name to use for conditioning
        
    Returns:
        Dictionary with conditional vectors
    """
    print("\n" + "="*60)
    print("CONDITIONAL VECTORS PREPARATION")
    print("="*60)
    
    # Select appropriate conditional column
    if conditional_column is None:
        # Look for a good categorical column for conditioning
        categorical_cols = data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            unique_count = data[col].nunique()
            if 2 <= unique_count <= 20:  # Good range for conditioning
                conditional_column = col
                break
    
    if conditional_column is None:
        print("   ❌ No suitable column found for conditioning")
        return None
    
    print(f"\n🎯 Using column for conditioning: '{conditional_column}'")
    
    # Get unique categories
    categories = data[conditional_column].unique()
    print(f"   📊 Number of categories: {len(categories)}")
    
    # Create one-hot encoded vectors
    cond_vectors = np.eye(len(categories))
    
    # Create mapping dictionary
    category_map = {cat: vec for cat, vec in zip(categories, cond_vectors)}
    
    print(f"\n🔢 Conditional vectors created:")
    for i, (category, vector) in enumerate(category_map.items()):
        print(f"   - {category}: {vector.tolist()}")
        if i >= 5:  # Show first 5 only
            print(f"   ... and {len(category_map) - 5} more")
            break
    
    return category_map

def visualize_key_distributions(data: pd.DataFrame) -> None:
    """
    Create visualizations for key variables in the dataset.
    
    Args:
        data: pandas DataFrame to visualize
    """
    print("\n" + "="*60)
    print("DATA VISUALIZATION")
    print("="*60)
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('SEER Orthopaedic-Oncology Dataset: Key Distributions', fontsize=16)
    
    # Plot 1: Survival months distribution
    if 'Survival months' in data.columns:
        axes[0, 0].hist(data['Survival months'], bins=50, alpha=0.7, color='skyblue')
        axes[0, 0].set_title('Survival Months Distribution')
        axes[0, 0].set_xlabel('Survival Months')
        axes[0, 0].set_ylabel('Frequency')
    
    # Plot 2: Age distribution
    age_col = 'Age recode with <1 year olds'
    if age_col in data.columns:
        age_counts = data[age_col].value_counts()
        axes[0, 1].bar(range(len(age_counts)), age_counts.values, color='lightgreen')
        axes[0, 1].set_title('Age Distribution')
        axes[0, 1].set_xlabel('Age Groups')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Plot 3: Top histology types
    if 'ICD-O-3 Hist/behav' in data.columns:
        top_histology = data['ICD-O-3 Hist/behav'].value_counts().head(10)
        axes[1, 0].barh(range(len(top_histology)), top_histology.values, color='salmon')
        axes[1, 0].set_title('Top 10 Histology Types')
        axes[1, 0].set_xlabel('Count')
        axes[1, 0].set_yticks(range(len(top_histology)))
        axes[1, 0].set_yticklabels([str(x)[:30] + '...' if len(str(x)) > 30 else str(x) 
                                   for x in top_histology.index])
    
    # Plot 4: Treatment distribution
    if 'Chemotherapy recode' in data.columns:
        treatment_counts = data['Chemotherapy recode'].value_counts()
        axes[1, 1].pie(treatment_counts.values, labels=treatment_counts.index, 
                      autopct='%1.1f%%', colors=['lightcoral', 'lightblue'])
        axes[1, 1].set_title('Chemotherapy Treatment Distribution')
    
    plt.tight_layout()
    plt.savefig('seer_dataset_overview.png', dpi=300, bbox_inches='tight')
    print("   📊 Visualizations saved to: seer_dataset_overview.png")
    plt.show()

def generate_summary_report(data: pd.DataFrame, metadata: SingleTableMetadata, 
                          conditional_vectors: Optional[dict]) -> None:
    """
    Generate a comprehensive summary report.
    
    Args:
        data: pandas DataFrame
        metadata: SDV metadata object
        conditional_vectors: Dictionary with conditional vectors
    """
    print("\n" + "="*60)
    print("SUMMARY REPORT")
    print("="*60)
    
    print(f"\n📋 Dataset Summary:")
    print(f"   - Total records: {len(data):,}")
    print(f"   - Total features: {data.shape[1]}")
    print(f"   - Numerical features: {len(data.select_dtypes(include=[np.number]).columns)}")
    print(f"   - Categorical features: {len(data.select_dtypes(include=['object']).columns)}")
    
    print(f"\n🎯 High-cardinality challenges:")
    categorical_cols = data.select_dtypes(include=['object']).columns
    high_card_cols = [col for col in categorical_cols if data[col].nunique() > 10]
    if high_card_cols:
        print(f"   - {len(high_card_cols)} columns with >10 unique values")
        for col in high_card_cols:
            print(f"     * {col}: {data[col].nunique():,} unique values")
    else:
        print("   - No high-cardinality columns")
    
    print(f"\n✅ Readiness for O2TAB-GAN:")
    print(f"   - Metadata schema: {'✅ Valid' if metadata else '❌ Invalid'}")
    print(f"   - Conditional vectors: {'✅ Prepared' if conditional_vectors else '❌ Not prepared'}")
    print(f"   - Data quality: {'✅ Good' if data.isnull().sum().sum() == 0 else '⚠️ Has missing values'}")
    
    print(f"\n🚀 Next Steps:")
    print(f"   1. Proceed to Phase 2: Architecture Implementation")
    print(f"   2. Fork CTAB-GAN+ repository")
    print(f"   3. Implement FT-Transformer and Fourier-feature components")

def main():
    """
    Main function to execute the data analysis pipeline.
    """
    print("="*60)
    print("O2TAB-GAN DATA ANALYSIS PIPELINE")
    print("="*60)
    
    # Step 1: Load data
    data_path = 'final_dataset.csv'
    data = load_and_inspect_data(data_path)
    if data is None:
        return
    
    # Step 2: Basic inspection
    basic_data_inspection(data)
    
    # Step 3: Analyze high-cardinality columns
    analyze_high_cardinality_columns(data)
    
    # Step 4: Create metadata schema
    metadata = create_metadata_schema(data)
    if metadata is None:
        return
    
    # Step 5: Prepare conditional vectors
    conditional_vectors = prepare_conditional_vectors(data)
    
    # Step 6: Visualize key distributions
    visualize_key_distributions(data)
    
    # Step 7: Generate summary report
    generate_summary_report(data, metadata, conditional_vectors)
    
    print(f"\n🎉 Phase 1 Task 2 (Data Analysis) completed successfully!")

if __name__ == "__main__":
    main() 