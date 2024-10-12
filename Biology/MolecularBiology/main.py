#%% Author - German Shiklov
# Date: 24.4.24
###########################
import abc
import random
import copy
from enum import Enum

#%%
class NucleotideGroup(Enum):
    NONE = '',
    PO4 = 'Phosphate',
    OH = 'Hydroxyl'


class Nucleotide:
    
    def __init__(self, data, group=NucleotideGroup.NONE):
        self.data = data
        self.next = None
        self.group = group
    
    @staticmethod
    def _FIVE_PRIME_():
        return Nucleotide("5'", NucleotideGroup.PO4)
    
    @staticmethod
    def _THREE_PRIME_():
        return Nucleotide("3'", NucleotideGroup.OH)

    @staticmethod
    def _is_equal_(this, other):
        if this is other:
            return True
        return (this.group == other.group and this.data == other.data)

class RibonucleicAcid(abc.ABC):
    def __init__(self) -> None:
        pass
    
    def complementary(self, nucleotide):
        return Nucleotide._THREE_PRIME_() if Nucleotide._is_equal_(nucleotide, Nucleotide._THREE_PRIME_) else Nucleotide._FIVE_PRIME_()
        

class RNA:    

    def __init__(self, is_leading_strand=True) -> None:
        self._rna_letters_2 = ['A', 'U']
        self._rna_letters_3 = ['C', 'G'] # Covallentic connections
        self._is_lead_strand = is_leading_strand
        self._strand = Nucleotide._FIVE_PRIME_() if is_leading_strand else Nucleotide._THREE_PRIME_()
        self._strand.next = Nucleotide._THREE_PRIME_() if is_leading_strand else Nucleotide._FIVE_PRIME_()
        # self.leading_strand = Nucleotide._FIVE_PRIME_()
        # self.leading_strand.next = Nucleotide._THREE_PRIME_()
        # self.lagging_strand = Nucleotide._THREE_PRIME_()
        # self.lagging_strand.next = Nucleotide._FIVE_PRIME_()

    def complementary(self, node):
        rna_list = self._rna_letters_2 if node.data in self._rna_letters_2 else self.dna_letters_3
        complementary_index = (self.rna_list.index(node.data) + 1)%2
        return Nucleotide(self._rna_letters_2[complementary_index])
    
    def insert(self, nucleotide):
        head = copy.deepcopy(self._strand)
        if self._is_lead_strand:
            while head.next.group != NucleotideGroup.OH:
                head = head.next
            nucleotide.next = head.next # assign last.
            head.next = nucleotide # update in between
            return True
        # Lagging Strand

class DNA:
    
    def __init__(self) -> None:
        self.dna_letters_2 = ['A', 'T']
        self.dna_letters_3 = ['C', 'G'] # Covallentic connections
        self.strand_A = Nucleotide._FIVE_PRIME_()
        self.strand_B = Nucleotide._THREE_PRIME_()

    def append(self, new_node):
        current = self.strand_A
        while current.next:
            current = current.next
        current.next = new_node

    def generate_strands(self, length):
        count_primers = 0
        should_region_primers = False
        dna_letters = self.dna_letters_2 + self.dna_letters_3
        for index in range(0, length):
            count_primers += 1
            if count_primers > int(0.1 * length): # Every once a while, prepare a rich area of A, T.
                prepare_4_primer = 0
                should_region_primers = not should_region_primers            
            node = Nucleotide(random.choice(self.dna_letters_2 + dna_letters if should_region_primers else dna_letters))
            self.strand_A.append(node)
            self.strand_B.append(self.complementary(node))
        self.strand_A.append(Nucleotide._THREE_PRIME_())
        self.strand_B.append(Nucleotide._FIVE_PRIME_())
    
    def complementary(self, node):
        dna_list = self.dna_letters_2 if node.data in self.dna_letters_2 else self.dna_letters_3
        complementary_index = (self.dna_list.index(node.data) + 1)%2
        return Nucleotide(self.dna_letters_2[complementary_index])
    
    def to_string(self):
        current = self.strand_A
        while current:
            print(current.data, end=" -> ")
            current = current.next
        print("None")

#%%
class Enzyme(abc.ABC):
    
    @abc.abstractmethod
    def run():
        return

class Primase(Enzyme):
    
    def __init__(self) -> None:
        self._rna_leading = RNA()
        self._rna_lagging = RNA()
        self._pol3 = Polymerase_3()
        pass

    def synthesize_leading_strand(self, nucleotide): # Attach Primers
        amount_to_synthesize = 10 + random.randint(0, 2)
        count = 0
        head_start = None
        current_nucleotide = copy.copy(nucleotide)
        if (nucleotide.group == NucleotideGroup.PO4):
            head_start = nucleotide
            nucleotide = nucleotide.next
        while (count < amount_to_synthesize and nucleotide.group != NucleotideGroup.OH):
            complementary_nucleotide = self._rna_leading.complementary(nucleotide)
            self._rna_leading.insert(complementary_nucleotide)

    def synthesize_lagging_strand():
        pass

class Topoisomerase(Enzyme):

    def run():
        pass

class Polymerase_1(Enzyme): # Prokrayot
    
    def __init__(self) -> None:
        pass

    def run():
        return
    
    def synthesize():
        pass
    
    def recombination_repair():
        pass
    
    def excision_repair():
        pass


class Polymerase_3(Enzyme):
    
    def __init__(self) -> None:
        pass

    def primer_synthesize_leading_strand(self, nucleotide):
        if (nucleotide.group == NucleotideGroup.PO4 and nucleotide.next.group == NucleotideGroup.OH):
            last_nucleotide = nucleotide.next
        for count in range(0, 10):
            pass
        while (nucleotide.group != NucleotideGroup.OH):
            pass
        

#%%

# %%
a =  Nucleotide("5'", NucleotideGroup.PO4)
b =  Nucleotide("5'", NucleotideGroup.PO4)
c = copy.deepcopy(b)
# a == b
b.next is c.next
# %%
b.next = a
# %%
a is b.next
# %%
