/-
Semantics of Spire's combinational operators on `BitVec`, mirroring `spire.simulator`.
Widths are explicit; operands are extended (or truncated) to the result width first, each by its
own signedness: S = sign-extend (`signExtend`, truncates when narrower), U = zero-extend (`setWidth`).
-/
namespace SpireSemantics

def resizeS {wa : Nat} (wo : Nat) (a : BitVec wa) : BitVec wo := a.signExtend wo
def resizeU {wa : Nat} (wo : Nat) (a : BitVec wa) : BitVec wo := a.setWidth wo

-- arithmetic; the two letters are the signedness of the left and right operand
def addSS {wa wb : Nat} (wo : Nat) (a : BitVec wa) (b : BitVec wb) : BitVec wo := (a.signExtend wo) + (b.signExtend wo)
def addSU {wa wb : Nat} (wo : Nat) (a : BitVec wa) (b : BitVec wb) : BitVec wo := (a.signExtend wo) + (b.setWidth wo)
def addUS {wa wb : Nat} (wo : Nat) (a : BitVec wa) (b : BitVec wb) : BitVec wo := (a.setWidth wo) + (b.signExtend wo)
def addUU {wa wb : Nat} (wo : Nat) (a : BitVec wa) (b : BitVec wb) : BitVec wo := (a.setWidth wo) + (b.setWidth wo)
def subSS {wa wb : Nat} (wo : Nat) (a : BitVec wa) (b : BitVec wb) : BitVec wo := (a.signExtend wo) - (b.signExtend wo)
def subSU {wa wb : Nat} (wo : Nat) (a : BitVec wa) (b : BitVec wb) : BitVec wo := (a.signExtend wo) - (b.setWidth wo)
def subUS {wa wb : Nat} (wo : Nat) (a : BitVec wa) (b : BitVec wb) : BitVec wo := (a.setWidth wo) - (b.signExtend wo)
def subUU {wa wb : Nat} (wo : Nat) (a : BitVec wa) (b : BitVec wb) : BitVec wo := (a.setWidth wo) - (b.setWidth wo)
def mulSS {wa wb : Nat} (wo : Nat) (a : BitVec wa) (b : BitVec wb) : BitVec wo := (a.signExtend wo) * (b.signExtend wo)
def mulSU {wa wb : Nat} (wo : Nat) (a : BitVec wa) (b : BitVec wb) : BitVec wo := (a.signExtend wo) * (b.setWidth wo)
def mulUS {wa wb : Nat} (wo : Nat) (a : BitVec wa) (b : BitVec wb) : BitVec wo := (a.setWidth wo) * (b.signExtend wo)
def mulUU {wa wb : Nat} (wo : Nat) (a : BitVec wa) (b : BitVec wb) : BitVec wo := (a.setWidth wo) * (b.setWidth wo)

-- bitwise: operands zero-extended to the result width (Spire's bitwise results are unsigned)
def bitAnd  {wa wb : Nat} (wo : Nat) (a : BitVec wa) (b : BitVec wb) : BitVec wo := (a.setWidth wo) &&& (b.setWidth wo)
def bitOr   {wa wb : Nat} (wo : Nat) (a : BitVec wa) (b : BitVec wb) : BitVec wo := (a.setWidth wo) ||| (b.setWidth wo)
def bitXor  {wa wb : Nat} (wo : Nat) (a : BitVec wa) (b : BitVec wb) : BitVec wo := (a.setWidth wo) ^^^ (b.setWidth wo)
def bitNand {wa wb : Nat} (wo : Nat) (a : BitVec wa) (b : BitVec wb) : BitVec wo := ~~~((a.setWidth wo) &&& (b.setWidth wo))
def bitNot  {wa : Nat} (a : BitVec wa) : BitVec wa := ~~~a

-- shifts: amount is the unsigned value of `b`; `<<` widens/truncates to wo first, `>>` is logical (also for signed)
def shl {wa wb : Nat} (wo : Nat) (a : BitVec wa) (b : BitVec wb) : BitVec wo := (a.setWidth wo) <<< b.toNat
def shr {wa wb : Nat} (wo : Nat) (a : BitVec wa) (b : BitVec wb) : BitVec wo := (a >>> b.toNat).setWidth wo

-- comparisons: exact integer compare, each operand read by its own signedness; 1-bit result
def toZ {w : Nat} (signed : Bool) (a : BitVec w) : Int := if signed then a.toInt else (a.toNat : Int)
def cmpEq {wa wb : Nat} (sa sb : Bool) (a : BitVec wa) (b : BitVec wb) : BitVec 1 := BitVec.ofBool (decide (toZ sa a = toZ sb b))
def cmpNe {wa wb : Nat} (sa sb : Bool) (a : BitVec wa) (b : BitVec wb) : BitVec 1 := BitVec.ofBool (decide (toZ sa a ≠ toZ sb b))
def cmpLt {wa wb : Nat} (sa sb : Bool) (a : BitVec wa) (b : BitVec wb) : BitVec 1 := BitVec.ofBool (decide (toZ sa a < toZ sb b))
def cmpLe {wa wb : Nat} (sa sb : Bool) (a : BitVec wa) (b : BitVec wb) : BitVec 1 := BitVec.ofBool (decide (toZ sa a ≤ toZ sb b))
def cmpGt {wa wb : Nat} (sa sb : Bool) (a : BitVec wa) (b : BitVec wb) : BitVec 1 := BitVec.ofBool (decide (toZ sa a > toZ sb b))
def cmpGe {wa wb : Nat} (sa sb : Bool) (a : BitVec wa) (b : BitVec wb) : BitVec 1 := BitVec.ofBool (decide (toZ sa a ≥ toZ sb b))

-- mux: `sel ≠ 0 ? a : b`; the caller has already resized both branches to the result width
def mux {ws wo : Nat} (sel : BitVec ws) (a b : BitVec wo) : BitVec wo := if sel = 0#ws then b else a

end SpireSemantics
