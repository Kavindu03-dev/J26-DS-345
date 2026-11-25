import "./globals.css";

export const metadata = {
  title: "C4 Tourism Agent – Demo",
  description: "Component 4 (J26-DS-345): LLM agent with function calling over C1 reviews, C2 hotels and C3 itineraries.",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
