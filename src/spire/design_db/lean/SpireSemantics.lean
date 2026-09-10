namespace SpireSemantics

def mulSS {wa wb : Nat} (wo : Nat) (a : BitVec wa) (b : BitVec wb) : BitVec wo :=
  (a.signExtend wo) * (b.signExtend wo)

def addSS {wa wb : Nat} (wo : Nat) (a : BitVec wa) (b : BitVec wb) : BitVec wo :=
  (a.signExtend wo) + (b.signExtend wo)

def resizeS {wa : Nat} (wo : Nat) (a : BitVec wa) : BitVec wo := a.signExtend wo
def resizeU {wa : Nat} (wo : Nat) (a : BitVec wa) : BitVec wo := a.setWidth wo

end SpireSemantics
