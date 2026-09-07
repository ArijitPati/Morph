import { Navbar } from "@/components/layout/Navbar";
import { Footer } from "@/components/layout/Footer";
import { PageContainer } from "@/components/layout/PageContainer";
import { Hero } from "@/components/landing/Hero";
import { CapabilitySection } from "@/components/landing/CapabilitySection";
import { HowMorphWorks } from "@/components/landing/HowMorphWorks";
import { TechStack } from "@/components/landing/TechStack";
import { SecuritySection } from "@/components/landing/SecuritySection";
import { ProblemStatement } from "@/components/landing/ProblemStatement";
import { FinalCTA } from "@/components/landing/FinalCTA";

export default function Home() {
  return (
    <>
      <Navbar />
      <PageContainer>
        <Hero />
        <CapabilitySection />
        <HowMorphWorks />
        <TechStack />
        <SecuritySection />
        <ProblemStatement />
        <FinalCTA />
      </PageContainer>
      <Footer />
    </>
  );
}
