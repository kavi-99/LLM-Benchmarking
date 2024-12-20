import AppMetricsPage from "../sections/@dashboard/app/AppMetricsPage";

export default function TimeBetweenToken() {
  return (
    <div>
      <div style={{ paddingBottom: "30px" }}>
        <AppMetricsPage metricType="timebetweentokens" title="Time Between Tokens Metrics" />
      </div>
      <div style={{ paddingBottom: "30px" }}>
        <AppMetricsPage metricType="timebetweentokens_median" title="Time Between Tokens Median Metrics" />
      </div>
      <div>
        <AppMetricsPage metricType="timebetweentokens_p95" title="Time Between Tokens P95 Metrics" />
      </div>
    </div>
  );
}
