import React from "react";
import { createRoot } from "react-dom/client";
import "@fontsource-variable/bricolage-grotesque";
import "@fontsource/atkinson-hyperlegible-next/400.css";
import "@fontsource/atkinson-hyperlegible-next/700.css";
import "./styles.css";
import App from "./App.jsx";

createRoot(document.getElementById("root")).render(<App />);
