

import DashboardLayout from "examples/LayoutContainers/DashboardLayout";
import DashboardNavbar from "examples/Navbars/DashboardNavbar";
import MDBox from "components/MDBox";
import MDTypography from "components/MDTypography";

import { useState } from "react";
import { Grid, Card, FormControl, InputLabel, Select, MenuItem, Button } from "@mui/material";
import ReportsLineChart from "examples/Charts/LineCharts/ReportsLineChart";

export default function Comparison() {
  const [selectedDataset, setSelectedDataset] = useState("Dataset A");
  const [selectedTarget, setSelectedTarget] = useState("target A");
  const datasets = ["Dataset A", "Dataset B", "Dataset C"];
  const targets = ["target A", "target B", "target C"];

  const models = ["NN", "QNN", "Mixture"];
  const hyperparams = ["Option 1", "Option 2", "Option 3"];

  return (
    <DashboardLayout>
      <DashboardNavbar />
      <MDBox py={3}>
        {/* Options Pane */}
        <Grid container spacing={3} mb={4}>
          <Grid item xs={12}>
            <FormControl size="small" sx={{ minWidth: 250 }}>
              <InputLabel>Select Dataset</InputLabel>
              <Select
                value={selectedDataset}
                label="Select Dataset"
                onChange={(e) => setSelectedDataset(e.target.value)}
              >
                {datasets.map((ds) => (
                  <MenuItem key={ds} value={ds}>
                    {ds}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>
        </Grid>
                <Grid container spacing={3} mb={4}>
          <Grid item xs={12}>
            <FormControl size="small" sx={{ minWidth: 250 }}>
              <InputLabel>Select Target</InputLabel>
              <Select
                value={selectedTarget}
                label="Select target"
                onChange={(e) => setSelectedTarget(e.target.value)}
              >
                {targets.map((ds) => (
                  <MenuItem key={ds} value={ds}>
                    {ds}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>
        </Grid>

              {/* <InputLabel>Select Target Column</InputLabel>
              <Select
                value={selectedTarget}
                label="Select Target Column"
                onChange={(e) => setSelectedTarget(e.target.value)}
              >
                {targets.map((ds) => (
                  <MenuItem key={ds} value={ds}>
                    {ds}
                  </MenuItem>
                ))}
              </Select> */}


        {/* Three Columns */}
        <Grid container spacing={3}>
          {models.map((model) => (
            <Grid item xs={12} md={4} key={model}>
              <Grid container justifyContent="space-between" alignItems="center" mb={2}>
                <Grid item>
                  <MDTypography variant="h6" fontWeight="bold">
                    {model}
                  </MDTypography>
                </Grid>
                <Grid item>
                  <Button variant="contained" size="small" color="white">Train & Predict</Button>
                </Grid>
              </Grid>


              {/* Hyperparameter Dropdowns */}
              <FormControl fullWidth size="small" sx={{ mb: 2 }}>
                <InputLabel>Hyperparam 1</InputLabel>
                <Select defaultValue={hyperparams[0]}>
                  {hyperparams.map((h) => (
                    <MenuItem key={h} value={h}>
                      {h}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>


              <FormControl fullWidth size="small" sx={{ mb: 3 }}>
                <InputLabel>Hyperparam 2</InputLabel>
                <Select defaultValue={hyperparams[1]}>
                  {hyperparams.map((h) => (
                    <MenuItem key={h} value={h}>
                      {h}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>


              {/* Three Segments (simple cards) */}
              {[1, 2, 3].map((seg) => (
                <Card key={`${model}-seg-${seg}`} sx={{ p: 2, mb: 2 }}>
                  <MDTypography variant="body2" color="text">
                    {model} Segment {seg} (placeholder)
                  </MDTypography>
                </Card>
              ))}
            </Grid>
          ))}
        </Grid>

        {/* Line Graph Container */}
        <MDBox mt={6}>
          <ReportsLineChart
            color="info"
            title="Performance Over Time"
            description="Line graph showing model performance trends"
            date="updated today"
            chart={{
              labels: ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul"],
              datasets: [
                { label: "NN", data: [50, 60, 65, 70, 75, 80, 85] },
                { label: "QNN", data: [45, 55, 60, 68, 72, 78, 82] },
                { label: "Mixture", data: [52, 62, 67, 73, 77, 83, 88] },
              ],
            }}
          />
        </MDBox>
      </MDBox>
    </DashboardLayout>
  );
}
