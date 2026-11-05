from databeacon.agent.verifier import extract_claims, verify_findings

EVIDENCE = [
    {"id": "E1", "tool": "run_sql", "ok": True, "data": {"rows": [[3, 946659.27], [4, 800218.73]]}},
    {"id": "E2", "tool": "period_comparison", "ok": True, "data": {"delta": -59195.34, "share": 40.42}},
]


def test_extract_ignores_years_quarters_and_dates():
    vals = [c.value for c in extract_claims("In Q4 2025 (2025-10-01 to 2025-12-31) revenue was $1.2M across 3 regions")]
    assert vals == [1_200_000]


def test_verified_direct_and_derived():
    f = [{"title": "a", "statement": "Revenue fell 15.5% from $946.7K to $800,219 [E1].", "evidence_ids": ["E1"]}]
    out = verify_findings(f, EVIDENCE)
    assert f[0]["verification"]["status"] == "verified"
    assert out["hallucination_rate"] == 0


def test_flags_fabricated_numbers():
    f = [{"title": "b", "statement": "Churn rose to 37.9% affecting 123,457 customers.", "evidence_ids": []},
         {"title": "c", "statement": "SMB West fell by 59,195 (40.4% of the decline).", "evidence_ids": ["E2"]}]
    out = verify_findings(f, EVIDENCE)
    assert f[0]["verification"]["status"] == "unverified"
    assert f[1]["verification"]["status"] == "verified"
    assert out["hallucination_rate"] == 0.5


def test_qualitative_and_missing_evidence():
    f = [{"title": "d", "statement": "Discounting appears to be the main driver.", "evidence_ids": ["E9"]}]
    verify_findings(f, EVIDENCE)
    assert f[0]["verification"]["status"] == "qualitative"
    assert f[0]["verification"]["missing_evidence_ids"] == ["E9"]
