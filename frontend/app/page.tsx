import Navbar from "@/components/landing/Navbar";
import Hero from "@/components/landing/Hero";
import FeaturesBento from "@/components/landing/FeaturesBento";
import PipelineShowcase from "@/components/landing/PipelineShowcase";

export default function LandingPage() {
  return (
    <>
      <Navbar />
      <main>
        <Hero />
        <FeaturesBento />
        <PipelineShowcase />
      </main>
    </>
  );
}
