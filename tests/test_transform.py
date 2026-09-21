import numpy as np

CORE=["plane_id","time_position","latitude","longitude","inside_saudi_boundary"]
EMISSIONS=["co2_g_s","nox_g_s","co_g_s","hc_g_s","sox_g_s","soot_g_s"]

def test_final_grain_and_core_keys(final_df):
    assert len(final_df)>0
    assert final_df.columns.duplicated().sum()==0
    assert final_df[CORE].notna().all().all()

def test_fuel_and_emission_eligibility(final_df):
    ready=final_df["fuel_flow_ready"].fillna(False).astype(bool)
    assert final_df.loc[ready,"fuel_flow_kg_s"].notna().all()
    assert (final_df.loc[ready,"fuel_flow_kg_s"]>0).all()
    assert final_df.loc[ready,EMISSIONS].notna().all().all()
    assert final_df.loc[~ready,EMISSIONS].isna().all().all()

def test_empty_input_edge_case(final_df):
    assert final_df.iloc[0:0].empty
